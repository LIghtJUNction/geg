#!/usr/bin/env python3
"""Live, local-model retrieval checks against geg's own implementation.

This is a small development regression set, not a general retrieval benchmark.
Uses current sources and function/body anchors, not file-name-only success.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

CASES = [
    ('主函数', 'src/main.rs', ['fn main()']),
    ('缓存的内容哈希在哪里生成', 'src/cache.rs', ['pub fn key(', 'hash.update(FORMAT.as_bytes())']),
    ('项目配置优先级怎么处理', 'src/config.rs', ['pub fn apply(', 'fn apply_file(']),
    ('如何选择GPU进行推理', 'src/embedding.rs', ['let gpu =', '.with_n_gpu_layers(']),
    ('文件内容如何分块', 'src/input.rs', ['pub fn chunks(']),
    ('终端退出后恢复键盘模式', 'src/tui.rs', ['disable_raw_mode']),
    ('删除全局缓存但是保留模型', 'src/cache_management.rs', ['DELETE FROM vectors_v1']),
    ('如何同时编码多个向量序列', 'src/embedding.rs', ['fn encode_batch(', 'pub fn embed_many(']),
    ('输出彩色结果', 'src/pretty.rs', ['let colored =', 'pub fn print(']),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='target/release/geg')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-entry', action='store_true', help='fail unless actual Rust main is in top 3')
    parser.add_argument('--min-hit-at-10', type=int, choices=range(len(CASES) + 1), default=0,
                        help='fail when fewer than this many implementation anchors reach top 10')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    binary = str(Path(args.binary).resolve(strict=True))
    results = []
    for query, filename, markers in CASES:
        source = (root / filename).read_text().splitlines()
        anchors = [i + 1 for i, line in enumerate(source) if any(m in line for m in markers)]
        if not anchors:
            raise AssertionError(f'regression anchor missing: {filename}: {markers}')
        command = [binary, '--no-config', '--no-progress', '--code', '--jsonl', query, '.']
        if args.model:
            command[1:1] = ['--model', str(args.model.resolve(strict=True))]
        started = time.monotonic()
        run = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=120)
        if run.returncode not in (0, 1):
            raise AssertionError(run.stderr)
        hits = [json.loads(line) for line in run.stdout.splitlines()]
        ranks = [i + 1 for i, h in enumerate(hits)
                 if (root / h['path']).resolve() == (root / filename).resolve()
                 and any(h['line'] <= line < h['line'] + len(h['text'].splitlines()) for line in anchors)]
        result = {'query': query, 'target': filename, 'rank': min(ranks, default=None),
                  'seconds': round(time.monotonic() - started, 3),
                  'top': [{'path': h['path'], 'line': h['line']} for h in hits[:3]]}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {f'hit_at_{k}': sum(r['rank'] is not None and r['rank'] <= k for r in results)
               for k in (1, 3, 10)}
    summary['queries'] = len(results)
    summary['note'] = 'Small local self-retrieval regression; warm caches are allowed; not a general quality claim.'
    print(json.dumps(summary, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps({'results': results, 'summary': summary}, ensure_ascii=False, indent=2) + '\n')
    if summary['hit_at_10'] < args.min_hit_at_10:
        raise SystemExit(f"retrieval gate failed: {summary['hit_at_10']}/{len(CASES)} < {args.min_hit_at_10}")
    if args.require_entry and (results[0]['rank'] is None or results[0]['rank'] > 3):
        raise SystemExit('actual Rust entry point missing from top 3')


if __name__ == '__main__':
    main()
