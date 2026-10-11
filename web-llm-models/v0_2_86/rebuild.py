"""Rebuild this release with explicitly selected TVM and MLC-LLM source trees."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--allow-different-sources", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "build-manifest.json").read_text())
    tvm = Path(os.environ["TVM_SOURCE_DIR"]).resolve()
    mlc = Path(os.environ["MLC_LLM_SOURCE_DIR"]).resolve()
    if not args.allow_different_sources:
        for name, directory in [("tvm", tvm), ("mlc_llm", mlc)]:
            revision = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
            if revision != manifest["sources"][name]["commit"]:
                raise ValueError(f"{directory} must be at {manifest['sources'][name]['commit']}")
            subprocess.run(["git", "-C", str(directory), "diff", "--exit-code", "HEAD"], check=True)
    env = dict(os.environ, TVM_HOME=str(tvm), SKIP_LOADING_MLCLLM_SO="1",
               PYTHONPATH=os.pathsep.join(map(str, [tvm / "python", tvm / ".local/python", mlc / "python"])),
               PYTHONNOUSERSITE="1", TVM_NUM_THREADS="4", BINARYEN_CORES="2")
    # The native compiler and the wasm bitcode must have been built from these trees.
    env.setdefault("TVM_LIBRARY_PATH", str(mlc / "build/lib"))
    env.setdefault("MLC_LIBRARY_PATH", str(mlc / "build"))

    def compile_library(library):
        config = ROOT / library["config"]
        if hashlib.sha256(config.read_bytes()).hexdigest() != library["config_sha256"]:
            raise ValueError(f"Config checksum mismatch: {config}")
        output = args.output / library["path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "mlc_llm", "compile", str(config),
                   *library["compile_args"], "--output", str(output)]
        with output.with_suffix(".log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(library["path"], hashlib.sha256(output.read_bytes()).hexdigest(), flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(compile_library, manifest["libraries"]))


if __name__ == "__main__":
    main()
