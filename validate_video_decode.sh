#!/usr/bin/env bash
# Linux CUDA validation + benchmark for the two upstream vLLM video-decode ports.
# Runs inside a Lepton batch job on one H100. Everything is logged to stdout and
# copied under $WORK/results. Exit status is non-zero if any test phase fails.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
WORK=${WORK:-/work}; RES=$WORK/results; mkdir -p "$RES"; cd "$WORK"
FORK=${FORK:-https://github.com/arif-ahmed-nv/vllm-windows.git}
BRANCH_TC=${BRANCH_TC:-feat/torchcodec-cuda-device}
BRANCH_NV=${BRANCH_NV:-feat/pynvvideocodec-in-memory}
BENCH_DIR=${BENCH_DIR:-$WORK/bench}
FAILURES=0
log() { echo; echo "===== [$(date -u +%H:%M:%S)] $*"; }
phase_result() { if [ "$1" -eq 0 ]; then echo "PHASE OK: $2"; else echo "PHASE FAILED($1): $2"; FAILURES=$((FAILURES+1)); fi; echo "$2 rc=$1" >> "$RES/phases.txt"; }

log "host"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true; nproc; free -g | sed -n '1,2p'
ls /usr/lib/x86_64-linux-gnu/libnvcuvid.so* 2>/dev/null || echo "WARN: libnvcuvid not visible; set NVIDIA_DRIVER_CAPABILITIES=compute,utility,video"

log "apt"; apt-get update -qq >/dev/null && apt-get install -y -qq --no-install-recommends git curl ca-certificates ffmpeg python3.12 python3.12-venv python3.12-dev build-essential >/dev/null
python3.12 -m venv "$WORK/venv" && "$WORK/venv/bin/pip" install -q -U pip uv
PYBIN=$WORK/venv/bin/python; UV="$WORK/venv/bin/uv pip install --python $PYBIN -q"

log "clone port branches"; git clone -q --depth 300 --branch "$BRANCH_TC" "$FORK" "$WORK/src-tc"; git clone -q --depth 300 --branch "$BRANCH_NV" "$FORK" "$WORK/src-nv"
for d in src-tc src-nv; do git -C "$WORK/$d" remote add upstream https://github.com/vllm-project/vllm.git; git -C "$WORK/$d" fetch -q --depth 300 upstream main; echo "$d: $(git -C "$WORK/$d" log -1 --format='%h %s') (base $(git -C "$WORK/$d" merge-base HEAD upstream/main | cut -c1-9))"; done

log "install vLLM nightly (brings the pinned torch)"; $UV --pre vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly 2>&1 | tail -3
$PYBIN -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
$UV setuptools setuptools-scm wheel packaging jinja2 cmake ninja pytest psutil 2>&1 | tail -1

# Editable install of a branch on top of the nightly compiled libs; fall back to
# overlaying the changed Python files onto the installed package.
install_branch() {
  local src=$1
  log "install branch $(git -C "$src" rev-parse --abbrev-ref HEAD) (precompiled editable)"
  if ( cd "$src" && VLLM_USE_PRECOMPILED=1 $UV --no-build-isolation -e . 2>&1 | tail -4 ) && $PYBIN -c "import vllm,os;assert os.path.realpath(os.path.dirname(vllm.__file__)).startswith(os.path.realpath('$src')), vllm.__file__" ; then
    echo "editable install OK -> $($PYBIN -c 'import vllm;print(vllm.__file__)')"; return 0
  fi
  echo "precompiled editable install failed; overlaying changed files onto the nightly package"
  $UV --pre vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly 2>&1 | tail -1
  local site; site=$($PYBIN -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
  local base; base=$(git -C "$src" merge-base HEAD upstream/main)
  for f in $(git -C "$src" diff --name-only "$base" HEAD -- 'vllm/*.py'); do cp -v "$src/$f" "$site/${f#vllm/}"; done
}

run_pytest() { # label, src, files..., -k expr via K env
  local label=$1 src=$2; shift 2
  log "pytest [$label]"
  ( cd "$src" && $PYBIN -m pytest -q --no-header -p no:cacheprovider -rfEs "$@" ${K:+-k "$K"} ) 2>&1 | tee "$RES/pytest-$label.txt" | tail -40
  phase_result "${PIPESTATUS[0]}" "pytest-$label"
}

TV=$($PYBIN -c "import torch;print(torch.__version__.split('+')[0])"); CU=$($PYBIN -c "import torch;print('cu'+torch.version.cuda.replace('.',''))")
log "torchcodec (CUDA build for torch $TV / $CU)"
echo "torch==$($PYBIN -c 'import torch;print(torch.__version__)')" > "$WORK/torch-constraint.txt"
if ! $UV torchcodec --index-url "https://download.pytorch.org/whl/$CU" --extra-index-url https://pypi.org/simple --constraint "$WORK/torch-constraint.txt" 2>&1 | tail -2; then
  echo "CUDA torchcodec unavailable for this torch; installing PyPI torchcodec (CPU-only tests will still run)"; $UV torchcodec 2>&1 | tail -1
fi
$PYBIN - <<'PY'
import torchcodec, torch
print("torchcodec", torchcodec.__version__)
try:
    from torchcodec.decoders import VideoDecoder
    import subprocess, tempfile, os
    p = tempfile.mktemp(suffix=".mp4")
    subprocess.run(["ffmpeg","-v","error","-y","-f","lavfi","-i","testsrc2=size=320x240:rate=30","-t","1","-c:v","libx264","-pix_fmt","yuv420p",p],check=True)
    d = VideoDecoder(open(p,"rb").read(), device="cuda"); f = d.get_frames_at([0]); print("torchcodec CUDA decode OK:", tuple(f.data.shape), f.data.device)
except Exception as e:
    print("torchcodec CUDA decode NOT available:", type(e).__name__, str(e)[:200])
PY

############ PR 1: TorchCodec device selection ############
install_branch "$WORK/src-tc"
K="torchcodec or backend_kwargs or device or lazy_imported or decoder_spec" run_pytest tc-video "$WORK/src-tc" tests/multimodal/test_video.py
K="" run_pytest tc-ipc "$WORK/src-tc" tests/multimodal/test_gpu_ipc_memory.py
K="gpu_video_backend" run_pytest tc-config "$WORK/src-tc" tests/config/test_multimodal_config.py
log "bench: opencv / torchcodec cpu / torchcodec cuda"
$PYBIN "$BENCH_DIR/bench_video_decode.py" clips --out "$WORK/clips" | tee "$RES/clips.txt"
for be in opencv torchcodec torchcodec-cuda; do $PYBIN "$BENCH_DIR/bench_video_decode.py" bench --backend "$be" --clips "$WORK/clips" --out "$RES/bench.jsonl" --label "tc-branch"; done
$PYBIN "$BENCH_DIR/bench_video_decode.py" correctness --backend torchcodec-cuda --clips "$WORK/clips" --out "$RES/bench.jsonl"

############ PR 2: PyNvVideoCodec in-memory input ############
install_branch "$WORK/src-nv"
for ver in 2.0.4 2.2.2; do
  log "PyNvVideoCodec==$ver"; $UV "PyNvVideoCodec==$ver" 2>&1 | tail -1; $PYBIN -c "import PyNvVideoCodec as n; print('PyNvVideoCodec', getattr(n,'__version__','?'))"
  K="pynvvideocodec" run_pytest "nv-video-$ver" "$WORK/src-nv" tests/multimodal/test_video.py
  modes="default"; [ "$ver" = "2.2.2" ] && modes="default forced-tempfile forced-tempfile-shm"
  for mode in $modes; do
    $PYBIN "$BENCH_DIR/bench_video_decode.py" bench --backend pynvvideocodec --pynv-mode "$mode" --clips "$WORK/clips" --out "$RES/bench.jsonl" --label "pynv-$ver-$mode"
  done
  $PYBIN "$BENCH_DIR/bench_video_decode.py" correctness --backend pynvvideocodec --clips "$WORK/clips" --out "$RES/bench.jsonl" --label "pynv-$ver"
  $PYBIN "$BENCH_DIR/bench_video_decode.py" ab-check --clips "$WORK/clips" --out "$RES/bench.jsonl" --label "pynv-$ver"; phase_result $? "ab-check-$ver"
done

log "REPORT"; $PYBIN "$BENCH_DIR/bench_video_decode.py" report --in "$RES/bench.jsonl" | tee "$RES/report.md"
log "phase summary"; cat "$RES/phases.txt"; echo "FAILED PHASES: $FAILURES"
exit $FAILURES
