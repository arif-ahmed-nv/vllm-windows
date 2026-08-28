# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared setuptools-rust build entry for Rust artifacts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from setuptools import setup
from setuptools_rust import Binding, RustExtension

ROOT_DIR = Path(__file__).resolve().parents[1]
RUST_BUILD_ENV_FILE = "VLLM_RUST_BUILD_ENV_FILE"


def rust_build_environment() -> dict[str, str] | None:
    """Load compiler overrides prepared by the Windows build script."""
    environment_path = os.environ.get(RUST_BUILD_ENV_FILE)
    if environment_path is None:
        return None

    overrides = json.loads(Path(environment_path).read_text(encoding="utf-8"))
    if not isinstance(overrides, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in overrides.items()
    ):
        raise ValueError(f"{RUST_BUILD_ENV_FILE} must contain a string mapping")

    environment = os.environ.copy()
    environment.update(overrides)
    return environment


def rust_extensions(*, optional: bool = False) -> list[RustExtension]:
    environment = rust_build_environment()
    return [
        RustExtension(
            target="vllm.vllm-rs",
            path="rust/src/cmd/Cargo.toml",
            args=["--bin", "vllm-rs"],
            features=["native-tls-vendored"],
            binding=Binding.Exec,
            optional=optional,
            env=environment,
        ),
        RustExtension(
            target="vllm._rust_tool_parser",
            path="rust/src/parser/python/Cargo.toml",
            features=["pyo3/abi3-py38"],
            binding=Binding.PyO3,
            optional=optional,
            py_limited_api=True,
            env=environment,
        ),
    ]


def rust_py_extension_module_names() -> list[str]:
    module_names = []
    for extension in rust_extensions():
        if extension.binding != Binding.PyO3:
            continue

        for target_name in extension.target.values():
            if target_name.startswith("vllm._rust_"):
                module_names.append(target_name.rsplit(".", 1)[-1])

    return module_names


def build_binary(build_rust_args: list[str]) -> None:
    os.chdir(ROOT_DIR)
    (ROOT_DIR / "vllm").mkdir(exist_ok=True)
    setup(
        name="vllm-rust-frontend-build",
        packages=[],
        rust_extensions=rust_extensions(optional=False),
        script_args=["build_rust", "--quiet", "--inplace", *build_rust_args],
    )


def main() -> None:
    build_binary(sys.argv[1:])


if __name__ == "__main__":
    main()
