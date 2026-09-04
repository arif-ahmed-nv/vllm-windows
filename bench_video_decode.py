#!/usr/bin/env python3
"""Benchmark, correctness and lifecycle checks for vLLM video decode backends.

Subcommands (each run in a fresh process so CUDA contexts and the PyNvVideoCodec
slot pool start clean):
  clips        generate synthetic H.264 test clips with ffmpeg
  bench        latency / throughput / CPU / GPU-memory for one backend
  correctness  sampled frames vs the OpenCV reference decode
  ab-check     PyNvVideoCodec single-slot A->B->A stale-decoder check
  report       aggregate results.jsonl into markdown tables
"""
import argparse, json, os, statistics, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CLIPS = {  # name: (size, fps, seconds, lavfi source)
    "1080p-10s": ("1920x1080", 30, 10, "testsrc2"),
    "2160p-10s": ("3840x2160", 30, 10, "testsrc2"),
    "1080p-60s": ("1920x1080", 30, 60, "testsrc2"),
    "720p-ab-b": ("1280x720", 30, 5, "smptebars"),  # distinct clip for the A->B check
}
FRAME_COUNTS = (8, 32)
CONCURRENCY = (1, 8)
ROUNDS = 5


def emit(out, rec):
    rec["ts"] = time.time()
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


def cmd_clips(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); manifest = {}
    for name, (size, fps, secs, src) in CLIPS.items():
        p = out / f"{name}.mp4"
        if not p.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"{src}=size={size}:rate={fps}",
                            "-t", str(secs), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-g", "60",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(p)], check=True)
        manifest[name] = {"path": str(p), "bytes": p.stat().st_size, "size": size, "fps": fps, "seconds": secs}
        print(f"{name}: {p.stat().st_size/1e6:.1f} MB")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def backend_kwargs(backend, pynv_mode="default"):
    if backend == "opencv":
        return {"backend": "opencv"}
    if backend == "torchcodec":
        return {"backend": "torchcodec"}
    if backend == "torchcodec-cuda":
        return {"backend": "torchcodec", "device": "cuda"}
    if backend == "pynvvideocodec":
        return {"backend": "pynvvideocodec", "hw_decoders": 2}
    raise SystemExit(f"unknown backend {backend}")


def apply_pynv_mode(mode):
    """Force the temporary-file path (optionally on tmpfs) to isolate the in-memory effect."""
    import vllm.multimodal.video_decoders.pynvvideocodec as nv
    supports = getattr(nv, "_pynvvideocodec_supports_in_memory_input", None)
    if mode.startswith("forced-tempfile") and supports is not None:
        nv._pynvvideocodec_supports_in_memory_input = lambda nvc: False
    if mode.endswith("-shm"):
        os.environ["TMPDIR"] = "/dev/shm"; tempfile.tempdir = None
    import PyNvVideoCodec as pv
    eff = "in-memory" if (supports is not None and mode == "default" and supports(pv)) else "tempfile"
    return getattr(pv, "__version__", "?"), eff, tempfile.gettempdir()


def gpu_mem_mib_for_pid(pid):
    try:
        outp = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout
        for line in outp.splitlines():
            p, m = [x.strip() for x in line.split(",")[:2]]
            if int(p) == pid:
                return int(m)
    except Exception:
        pass
    return None


def cmd_bench(a):
    import psutil
    from vllm.multimodal.video import VideoBackend
    kw = backend_kwargs(a.backend); extra = {}
    if a.backend == "pynvvideocodec":
        ver, eff, tmp = apply_pynv_mode(a.pynv_mode); extra = {"pynv_version": ver, "pynv_path": eff, "tmpdir": tmp, "pynv_mode": a.pynv_mode}
    manifest = json.loads((Path(a.clips) / "manifest.json").read_text())
    proc = psutil.Process(); pid = os.getpid()
    for clip in ("1080p-10s", "2160p-10s", "1080p-60s"):
        data = Path(manifest[clip]["path"]).read_bytes()
        for nf in FRAME_COUNTS:
            VideoBackend.load_bytes(data, num_frames=nf, **kw)  # warm-up (decoder/context init)
            for conc in CONCURRENCY:
                lat, walls = [], []
                cpu0 = proc.cpu_times(); t_all0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=conc) as ex:
                    for _ in range(ROUNDS):
                        t0 = time.perf_counter()
                        def one():
                            s = time.perf_counter(); f, m = VideoBackend.load_bytes(data, num_frames=nf, **kw)
                            assert f.shape[0] == len(m["frames_indices"]) == nf, f.shape
                            return time.perf_counter() - s
                        lat += list(ex.map(lambda _: one(), range(conc)))
                        walls.append(time.perf_counter() - t0)
                t_all = time.perf_counter() - t_all0; cpu1 = proc.cpu_times()
                cpu_s = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
                emit(a.out, {"kind": "bench", "label": a.label, "backend": a.backend, "clip": clip, "num_frames": nf,
                             "concurrency": conc, "rounds": ROUNDS, "lat_ms_p50": statistics.median(lat) * 1e3,
                             "lat_ms_p95": sorted(lat)[max(0, int(len(lat) * 0.95) - 1)] * 1e3,
                             "req_per_s": (ROUNDS * conc) / t_all, "frames_per_s": (ROUNDS * conc * nf) / t_all,
                             "cpu_cores_avg": cpu_s / t_all, "gpu_mem_mib_proc": gpu_mem_mib_for_pid(pid), **extra})


def frame_diff(ref, test):
    import numpy as np
    d = np.abs(ref.astype(np.int16) - test.astype(np.int16))
    return {"mae": float(d.mean()), "p99": float(np.percentile(d, 99)), "max": int(d.max())}


def cmd_correctness(a):
    from vllm.multimodal.video import VideoBackend
    kw = backend_kwargs(a.backend); extra = {}
    if a.backend == "pynvvideocodec":
        ver, eff, _ = apply_pynv_mode("default"); extra = {"pynv_version": ver, "pynv_path": eff}
    manifest = json.loads((Path(a.clips) / "manifest.json").read_text())
    for clip in ("1080p-10s", "2160p-10s", "720p-ab-b"):
        data = Path(manifest[clip]["path"]).read_bytes()
        ref, rm = VideoBackend.load_bytes(data, num_frames=8, backend="opencv")
        got, gm = VideoBackend.load_bytes(data, num_frames=8, **kw)
        rec = {"kind": "correctness", "label": a.label, "backend": a.backend, "clip": clip, "shape_match": tuple(ref.shape) == tuple(got.shape),
               "indices_match": list(rm["frames_indices"]) == list(gm["frames_indices"]), "indices": list(gm["frames_indices"]), **extra}
        if rec["shape_match"]:
            rec.update(frame_diff(ref, got))
        emit(a.out, rec)


def cmd_ab_check(a):
    """hw_decoders=1 so A and B must share one slot; B must not return stale A data."""
    from vllm.multimodal.video import VideoBackend
    import vllm.multimodal.video_decoders.pynvvideocodec as nv
    ver, eff, _ = apply_pynv_mode("default")
    manifest = json.loads((Path(a.clips) / "manifest.json").read_text())
    A = Path(manifest["1080p-10s"]["path"]).read_bytes(); B = Path(manifest["720p-ab-b"]["path"]).read_bytes()
    refA, _ = VideoBackend.load_bytes(A, num_frames=8, backend="opencv"); refB, _ = VideoBackend.load_bytes(B, num_frames=8, backend="opencv")
    kw = {"backend": "pynvvideocodec", "hw_decoders": 1}
    seq, ok = [], True
    for name, data, ref in (("A", A, refA), ("B", B, refB), ("A2", A, refA)):
        got, m = VideoBackend.load_bytes(data, num_frames=8, **kw)
        slots = nv._pynv_decoder_pool.slots
        dec_id = id(slots[0].decoder) if slots and slots[0].decoder is not None else None
        d = frame_diff(ref, got) if got.shape == ref.shape else {"mae": None}
        step = {"step": name, "shape": list(got.shape), "expected_shape": list(ref.shape), "indices": m["frames_indices"],
                "slot_count": len(slots), "slot_id": id(slots[0]) if slots else None, "stream_id": id(slots[0].stream) if slots else None,
                "decoder_id": dec_id, **d}
        step["pass"] = got.shape == ref.shape and d.get("mae") is not None and d["mae"] < 3.0
        ok &= step["pass"]; seq.append(step)
    same_slot = len({s["slot_id"] for s in seq}) == 1 and len({s["stream_id"] for s in seq}) == 1
    decoder_rebuilt_for_B = seq[0]["decoder_id"] != seq[1]["decoder_id"]
    emit(a.out, {"kind": "ab-check", "label": a.label, "pynv_version": ver, "pynv_path": eff, "single_slot_retained": same_slot,
                 "decoder_rebuilt_for_B": decoder_rebuilt_for_B, "steps": seq, "pass": bool(ok and same_slot)})
    sys.exit(0 if ok and same_slot else 1)


def cmd_report(a):
    recs = [json.loads(l) for l in open(a.inp) if l.strip()]
    print("## Bench (median latency ms | p95 | frames/s | avg CPU cores | proc GPU MiB)\n")
    print("| label | backend | path | clip | frames | conc | p50 ms | p95 ms | frames/s | cpu cores | GPU MiB |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for r in [x for x in recs if x["kind"] == "bench"]:
        print(f"| {r['label']} | {r['backend']} | {r.get('pynv_path','-')} | {r['clip']} | {r['num_frames']} | {r['concurrency']} | {r['lat_ms_p50']:.0f} | {r['lat_ms_p95']:.0f} | {r['frames_per_s']:.1f} | {r['cpu_cores_avg']:.2f} | {r.get('gpu_mem_mib_proc') or '-'} |")
    print("\n## Correctness vs OpenCV (8 sampled frames)\n\n| label | backend | clip | shape | indices | MAE | p99 | max |\n|---|---|---|---|---|---|---|---|")
    for r in [x for x in recs if x["kind"] == "correctness"]:
        print(f"| {r['label']} | {r['backend']} | {r['clip']} | {r['shape_match']} | {r['indices_match']} | {r.get('mae','-') if r.get('mae') is None else round(r['mae'],3)} | {r.get('p99','-')} | {r.get('max','-')} |")
    print("\n## PyNvVideoCodec single-slot A->B->A check\n")
    for r in [x for x in recs if x["kind"] == "ab-check"]:
        print(f"- {r['label']} ({r['pynv_version']}, {r['pynv_path']}): PASS={r['pass']} single_slot_retained={r['single_slot_retained']} decoder_rebuilt_for_B={r['decoder_rebuilt_for_B']}")
        for s in r["steps"]:
            print(f"    - {s['step']}: shape={s['shape']} expected={s['expected_shape']} mae={s.get('mae')} pass={s['pass']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("clips"); s.add_argument("--out", required=True); s.set_defaults(fn=cmd_clips)
    s = sub.add_parser("bench"); s.add_argument("--backend", required=True); s.add_argument("--pynv-mode", default="default"); s.add_argument("--clips", required=True); s.add_argument("--out", required=True); s.add_argument("--label", default=""); s.set_defaults(fn=cmd_bench)
    s = sub.add_parser("correctness"); s.add_argument("--backend", required=True); s.add_argument("--clips", required=True); s.add_argument("--out", required=True); s.add_argument("--label", default=""); s.set_defaults(fn=cmd_correctness)
    s = sub.add_parser("ab-check"); s.add_argument("--clips", required=True); s.add_argument("--out", required=True); s.add_argument("--label", default=""); s.set_defaults(fn=cmd_ab_check)
    s = sub.add_parser("report"); s.add_argument("--in", dest="inp", required=True); s.set_defaults(fn=cmd_report)
    args = ap.parse_args(); args.fn(args)
