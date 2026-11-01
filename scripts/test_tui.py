#!/usr/bin/env python3
"""Local Unix PTY regression: keyboard, Unicode, preview, selection and terminal cleanup.

Usage: python3 scripts/test_tui.py --binary target/release/geg
No model, GPU, GUI automation or third-party Python packages required.
"""
import argparse
import errno
import fcntl
import os
from pathlib import Path
import pty
import select
import signal
import struct
import tempfile
import termios
import time


def scenario(binary, root, select_result):
    output_r, output_w = os.pipe()
    pid, master = pty.fork()
    if pid == 0:
        os.close(output_r)
        os.dup2(output_w, 1)
        os.close(output_w)
        os.chdir(root)
        os.environ['HOME'] = str(root)
        os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
        os.execv(binary, [binary, '--no-config', '--model', str(root / 'missing.gguf'),
                          '--no-progress'])
    os.close(output_w)
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack('HHHH', 24, 100, 0, 0))
    os.kill(pid, signal.SIGWINCH)
    screen, output = bytearray(), bytearray()
    status = None

    def pump(seconds):
        nonlocal status
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            for fd in select.select([master, output_r], [], [], 0.02)[0]:
                try:
                    data = os.read(fd, 65536)
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
                    data = b''
                if data:
                    (screen if fd == master else output).extend(data)
            if status is None:
                ended, child_status = os.waitpid(pid, os.WNOHANG)
                if ended:
                    status = os.waitstatus_to_exitcode(child_status)

    try:
        for _ in range(50):
            pump(0.1)
            if b'fuzzy file picker' in screen or status is not None:
                break
        assert b'fuzzy file picker' in screen, (status, screen[-2000:], output[-2000:])
        if select_result:
            os.write(master, '中文'.encode())
            pump(0.1)
            os.write(master, b'\x15rtry')  # Ctrl-U, fuzzy abbreviation
            pump(0.5)  # semantic worker reports missing model, fuzzy picker stays usable
            os.write(master, b'\t')  # preview on
            pump(0.1)
            assert b'retry_request' in screen, screen[-2000:]
            os.write(master, b'\r')
        else:
            os.write(master, b'\x1b')
        pump(0.5)
        assert status == (0 if select_result else 1), (status, screen[-3000:])
        lflag = termios.tcgetattr(master)[3]
        assert lflag & termios.ICANON and lflag & termios.ECHO, 'raw mode not restored'
        if select_result:
            assert output.count(b'| path | line | score | text |') == 1, output
            assert b'src/retry.rs' in output and b'\x1b' not in output, output
        else:
            assert not output, output
    finally:
        if status is None:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        os.close(master)
        os.close(output_r)


def real_model_scenario(binary, model, repository):
    """Two queries in one retained GPU context, then exit without a teardown race."""
    with tempfile.TemporaryDirectory(prefix='geg-tui-real-') as temporary:
        read_fd, write_fd = os.pipe()
        pid, master = pty.fork()
        if pid == 0:
            os.close(read_fd)
            os.dup2(write_fd, 1)
            os.close(write_fd)
            os.chdir(repository)
            os.execv(binary, [binary, '--no-config', '--model', model,
                '--cache', temporary + '/vectors.db', '--device', 'gpu',
                '--code', '--jsonl'])
        os.close(write_fd)
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack('HHHH', 30, 150, 0, 0))
        screen, output = bytearray(), bytearray()
        status = None

        def wait_for_result():
            started = time.monotonic()
            offset = len(screen)
            while time.monotonic() - started < 60:
                for fd in select.select([master, read_fd], [], [], 0.02)[0]:
                    try:
                        data = os.read(fd, 65536)
                    except OSError as exc:
                        if exc.errno != errno.EIO:
                            raise
                        data = b''
                    (screen if fd == master else output).extend(data)
                if b' semantic results' in screen[offset:]:
                    return time.monotonic() - started
            raise AssertionError('semantic timeout: ' + screen[-1000:].decode(errors='replace'))

        try:
            deadline = time.monotonic() + 10
            while b'fuzzy file picker' not in screen and time.monotonic() < deadline:
                if select.select([master], [], [], 0.05)[0]:
                    screen.extend(os.read(master, 65536))
            assert b'fuzzy file picker' in screen, screen[-1000:]
            os.write(master, '失败请求的重试逻辑'.encode())
            first = wait_for_result()
            os.write(master, b'\x15' + '鉴权失败后的处理'.encode())
            second = wait_for_result()
            os.write(master, b'\t')
            time.sleep(0.1)
            os.write(master, b'\r')
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                for fd in select.select([master, read_fd], [], [], 0.02)[0]:
                    try:
                        data = os.read(fd, 65536)
                    except OSError:
                        data = b''
                    (screen if fd == master else output).extend(data)
                ended, code = os.waitpid(pid, os.WNOHANG)
                if ended:
                    status = os.waitstatus_to_exitcode(code)
                    break
            assert status == 0, ('GPU TUI exit status', status)
            import json
            hit = json.loads(output)
            assert hit['score'] > 0 and (Path(repository) / hit['path']).is_file(), hit
            lflag = termios.tcgetattr(master)[3]
            assert lflag & termios.ICANON and lflag & termios.ECHO
            print(json.dumps({'gpu_tui_first_s': first, 'gpu_tui_second_s': second,
                'selected_path': hit['path'], 'terminal_restored': True}, ensure_ascii=False))
        finally:
            if status is None:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            os.close(master)
            os.close(read_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='target/release/geg')
    parser.add_argument('--model', type=Path, help='optional real GPU multi-query regression; also requires --repository')
    parser.add_argument('--repository', type=Path)
    args = parser.parse_args()
    if bool(args.model) != bool(args.repository):
        parser.error('--model and --repository must be supplied together')
    binary = str(Path(args.binary).resolve(strict=True))
    with tempfile.TemporaryDirectory(prefix='geg-tui-test-') as temporary:
        root = Path(temporary)
        (root / 'src').mkdir()
        (root / 'src' / 'retry.rs').write_text('fn retry_request() { /* retry with backoff */ }\n')
        (root / 'src' / 'auth.rs').write_text('fn authenticate() {}\n')
        (root / '.gitignore').write_text('src/ignored.rs\n')
        (root / 'src' / 'ignored.rs').write_text('ignored\n')
        blocked = root / 'src' / 'restricted'
        blocked.mkdir()
        (blocked / 'secret.rs').write_text('must not be scanned\n')
        blocked.chmod(0)
        try:
            scenario(binary, root, False)
            scenario(binary, root, True)
        finally:
            blocked.chmod(0o700)
    print('PASS: PTY Escape/selection, Unicode edits, fuzzy filtering, Tab preview, clean stdout, terminal restoration')
    if args.model:
        real_model_scenario(binary, str(args.model.resolve(strict=True)), str(args.repository.resolve(strict=True)))


if __name__ == '__main__':
    main()
