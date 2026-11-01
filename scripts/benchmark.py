#!/usr/bin/env python3
"""Local, real-model smoke benchmark; Linux/macOS, Python stdlib only.

Run: python3 scripts/benchmark.py --binary target/release/geg --model /path/model.gguf
The fixture has 100 small files; timings are not large-repository benchmarks.
Cold means empty geg vector cache, not an evicted OS filesystem page cache.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time


def measure_worker(command):
    import resource
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform == "darwin":
        rss /= 1024
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    records = [json.loads(line) for line in result.stdout.splitlines()]
    print(json.dumps({"seconds": elapsed, "peak_rss_mib": rss / 1024,
                      "records": len(records), "first": records[0] if records else None}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="target/release/geg")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    binary = str(Path(args.binary).resolve(strict=True))
    model = str(Path(args.model).resolve(strict=True))
    with tempfile.TemporaryDirectory(prefix="geg-bench-") as temporary:
        root = Path(temporary)
        corpus = root / "src"
        corpus.mkdir()
        for i in range(100):
            (corpus / f"module_{i:03}.txt").write_text(
                f"Module {i}: passwords are verified with argon2 before login.\n"
                f"Module {i}: network requests retry with exponential backoff.\n"
                f"Module {i}: the garden has trees and flowers.\n", encoding="utf-8")
        common = [binary, "--model", model, "--cache", str(root / "vectors.db"), "--no-index-hint"]
        query = ["--json", "--code", "password verification", str(corpus)]

        def measure(options):
            result = subprocess.run([sys.executable, __file__, "--measure", *common, *options],
                                    capture_output=True, text=True, check=True)
            return json.loads(result.stdout)

        cold = measure(query)
        warm = [measure(query) for _ in range(5)]
        index = root / "index.db"
        build = measure(["index", str(corpus), "--output", str(index)])
        rebuild = measure(["index", str(corpus), "--output", str(index)])
        indexed = [measure(["query", "password verification", "--index", str(index), "--code", "--json"]) for _ in range(5)]
        print(json.dumps({"fixture": "100 files, 300 lines, 100 default-size chunks",
                          "cold_definition": "empty vector cache; OS page cache not evicted",
                          "model_bytes": Path(model).stat().st_size,
                          "binary_bytes": Path(binary).stat().st_size,
                          "cold_search": cold,
                          "warm_search_median_s": statistics.median(x["seconds"] for x in warm),
                          "warm_peak_rss_mib": max(x["peak_rss_mib"] for x in warm),
                          "index_build": build,
                          "index_rebuild": rebuild,
                          "indexed_query_median_s": statistics.median(x["seconds"] for x in indexed),
                          "indexed_peak_rss_mib": max(x["peak_rss_mib"] for x in indexed)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--measure":
        measure_worker(sys.argv[2:])
    else:
        main()
