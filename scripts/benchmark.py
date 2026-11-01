#!/usr/bin/env python3
"""Local real-model benchmark; Python stdlib only, Linux/macOS.

Cold = an empty isolated vector database, NOT evicted OS/driver shader caches.
A supplied repository is never modified. Full indexing is opt-in for repositories.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time


def measure_worker():
    import resource
    request = json.load(sys.stdin)
    started = time.perf_counter()
    result = subprocess.run(request['command'], input=request.get('input'),
                            cwd=request.get('cwd'), capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform == 'darwin':
        rss /= 1024
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    records = [json.loads(line) for line in result.stdout.splitlines()]
    response = records[0] if len(records) == 1 else {}
    if 'error' in response:
        raise RuntimeError(response['error'])
    hits = response.get('hits', records)
    print(json.dumps({'seconds': elapsed, 'peak_rss_mib': rss / 1024,
                      'records': len(hits), 'first': hits[0] if hits else None,
                      'stats': response.get('stats'),
                      'device_log': [line for line in result.stderr.splitlines()
                                     if 'inference' in line and 'using ' in line]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='target/release/geg')
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', choices=['auto', 'gpu', 'cpu'], default='auto')
    parser.add_argument('--repository', type=Path)
    parser.add_argument('--query', default='password verification')
    parser.add_argument('--warm-runs', type=int, default=3)
    parser.add_argument('--with-index', action='store_true',
                        help='also build a complete repository index (may be slow)')
    args = parser.parse_args()
    if args.warm_runs < 1:
        parser.error('--warm-runs must be positive')
    binary = str(Path(args.binary).resolve(strict=True))
    model = str(Path(args.model).resolve(strict=True))
    with tempfile.TemporaryDirectory(prefix='geg-bench-') as temporary:
        root = Path(temporary)
        corpus = args.repository.resolve(strict=True) if args.repository else root / 'src'
        if not args.repository:
            corpus.mkdir()
            for i in range(100):
                (corpus / f'module_{i:03}.txt').write_text(
                    f'Module {i}: passwords are verified with argon2 before login.\n'
                    f'Module {i}: network requests retry with exponential backoff.\n'
                    f'Module {i}: the garden has trees and flowers.\n', encoding='utf-8')
        common = [binary, '--no-config', '--model', model, '--device', args.device,
                  '--cache', str(root / 'vectors.db'), '--no-index-hint']

        def measure(options, input_text=None):
            request = {'command': common + options, 'input': input_text, 'cwd': str(root)}
            result = subprocess.run([sys.executable, __file__, '--measure'],
                                    input=json.dumps(request), capture_output=True,
                                    text=True, check=False)
            if result.returncode:
                raise RuntimeError(result.stderr)
            return json.loads(result.stdout)

        options = ['--interactive', '--code', str(corpus)]
        cold = measure(options, args.query + '\n')
        warm = [measure(options, args.query + '\n') for _ in range(args.warm_runs)]
        report = {'fixture': str(corpus) if args.repository else '100 synthetic files / 300 lines',
                  'query': args.query, 'device': args.device,
                  'cold_definition': 'empty isolated vector DB; OS page cache and GPU shader cache not evicted',
                  'model_bytes': Path(model).stat().st_size,
                  'binary_bytes': Path(binary).stat().st_size,
                  'cold_search': cold,
                  'warm_search_median_s': statistics.median(x['seconds'] for x in warm),
                  'warm_peak_rss_mib': max(x['peak_rss_mib'] for x in warm),
                  'warm_stats': warm[0]['stats']}
        if not args.repository or args.with_index:
            index = root / 'index.db'
            report['index_build'] = measure(['index', str(corpus), '--output', str(index)])
            report['index_rebuild'] = measure(['index', str(corpus), '--output', str(index)])
            indexed = [measure(['query', args.query, '--index', str(index), '--code', '--jsonl'])
                       for _ in range(args.warm_runs)]
            report['indexed_query_median_s'] = statistics.median(x['seconds'] for x in indexed)
            report['indexed_peak_rss_mib'] = max(x['peak_rss_mib'] for x in indexed)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--measure':
        measure_worker()
    else:
        main()
