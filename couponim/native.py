"""Build and invoke the optional C++17 kernels with CPython RNG states."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

from .rr import CouponRR

CPP = Path(__file__).parent / "cpp"


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def executable(build_dir, name):
    return Path(build_dir).resolve() / (name + (".exe" if os.name == "nt" else ""))


def build(build_dir, compiler=None):
    """GCC/Clang convenience builder; CMake is also supported."""
    compiler = compiler or os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise RuntimeError("C++17 compiler not found. Install GCC/Clang, set CXX, or use --backend python")
    compiler = shutil.which(compiler) or str(Path(compiler).resolve())
    folder = Path(build_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    flags = ["-O3", "-std=c++17", "-ffp-contract=off"]
    if os.name == "nt":
        flags += ["-static-libgcc", "-static-libstdc++"]
    version = subprocess.run([compiler, "--version"], check=True, capture_output=True, text=True).stdout
    signature = {"compiler": compiler, "version": version, "flags": flags,
                 "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(CPP.iterdir()) if p.suffix in {".cpp", ".hpp"}}}
    record = folder / "build.json"
    if record.exists() and json.loads(record.read_text()) == signature and all(
            executable(folder, n).exists() for n in ("coupon_stream", "coupon_experiment")):
        return signature
    for name, source in [("coupon_stream", "streaming_main.cpp"),
                         ("coupon_experiment", "experiment_main.cpp")]:
        command = [compiler, *flags, str(CPP / source), "-o", str(executable(folder, name))]
        if os.name == "nt" and name == "coupon_stream":
            command.append("-lpsapi")
        subprocess.run(command, check=True)
    record.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    return signature


def dump_state(stream, seed):
    if sys.implementation.name != "cpython":
        raise RuntimeError("The native RNG bridge requires CPython")
    state = random.Random(int(seed)).getstate()
    if state[0] != 3 or len(state[1]) != 625:
        raise RuntimeError("Unsupported CPython random state format")
    np.asarray(state[1], dtype="<u4").tofile(stream)


def write_graph(path, sampler):
    if sys.byteorder != "little":
        raise RuntimeError("The native binary bridge currently supports little-endian hosts")
    g = sampler.graph
    with Path(path).open("wb") as stream:
        stream.write(struct.pack("<II", g.n, g.m))
        for data, dtype in [(g.indptr, "<u8"), (g.indices, "<i4"),
                            (sampler.alpha, "<f8"), (sampler.discard, "<f8"),
                            (sampler.inptr, "<u8"), (sampler.inedges, "<u8"),
                            (sampler.sources, "<i4"), (sampler.low, "<f8"), (sampler.high, "<f8")]:
            np.asarray(data, dtype=dtype).tofile(stream)


class NativeModel:
    """Uniform-neighbor forward model; preparation is timed separately."""
    def __init__(self, graph, alpha, discard, build_dir, work):
        self.graph = graph
        self.build_dir = Path(build_dir).resolve()
        self.work = Path(work).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.graphfile = self.work / "graph.bin"
        started = time.perf_counter()
        write_graph(self.graphfile, CouponRR(graph, alpha, discard))
        self.preparation_seconds = time.perf_counter() - started

    def stream(self, k, theta, root_seed, component_seeds):
        if not 1 <= k <= self.graph.n or not 0 < theta < 2**32 or len(component_seeds) != k:
            raise ValueError("Invalid native SRR dimensions")
        inp = self.work / "selection-input.bin"
        key = hashlib.sha256(json.dumps([k, theta, root_seed, list(component_seeds),
                                        file_digest(self.graphfile)]).encode()).hexdigest()[:20]
        output = self.work / f"native-selection-{key}"
        final = self.work / f"native-selection-{key}.k{k}.json"
        # The scheduler validates the protocol before reusing this checkpoint.
        if not final.exists():
            with inp.open("wb") as stream:
                stream.write(struct.pack("<II", theta, k))
                dump_state(stream, root_seed)
                for seed in component_seeds:
                    dump_state(stream, seed)
            subprocess.run([str(executable(self.build_dir, "coupon_stream")), str(self.graphfile),
                            str(inp), str(output), str(k)], check=True)
        result = json.loads(final.read_text(encoding="utf-8"))
        result["training_estimate_not_quality"] = self.graph.n * result["covered_count"] / theta
        result["selection_seconds"] = result["selection_elapsed_seconds"]
        return result.pop("selected"), result

    def forward_bank(self, batches, seed):
        if not 0 < batches < 2**32:
            raise ValueError("Invalid forward batch count")
        inp = self.work / "forward-input.bin"
        target = self.work / "forward-bank.bin"
        with inp.open("wb") as stream:
            stream.write(struct.pack("<I", batches))
            dump_state(stream, seed)
        subprocess.run([str(executable(self.build_dir, "coupon_experiment")), "forward",
                        str(self.graphfile), str(inp), str(target)], check=True)
        bank = []
        with target.open("rb") as stream:
            for _ in range(self.graph.n):
                length, = struct.unpack("<I", stream.read(4))
                values = np.fromfile(stream, dtype="<i8", count=length)
                if len(values) != length:
                    raise ValueError("Truncated native forward bank")
                bank.append(values)
        return bank

    def evaluate(self, seeds, trials, seed):
        if not 0 < trials < 2**32 or len(set(seeds)) != len(seeds) or not seeds:
            raise ValueError("Invalid evaluation budget or seeds")
        if any(not 0 <= s < self.graph.n for s in seeds):
            raise ValueError("Invalid evaluation node")
        inp = self.work / "evaluation-input.bin"
        target = self.work / "evaluation.json"
        with inp.open("wb") as stream:
            stream.write(struct.pack("<III", 1, trials, len(seeds)))
            np.asarray(seeds, dtype="<i4").tofile(stream)
            dump_state(stream, seed)
        subprocess.run([str(executable(self.build_dir, "coupon_experiment")), "evaluate",
                        str(self.graphfile), str(inp), str(target)], check=True)
        return json.loads(target.read_text(encoding="utf-8"))[0]
