"""Offline config/report and mocked Linux worker checks; no real training."""
import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fs_screen', HERE / 'screen.py')
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def posix(path):
    value = str(Path(path).resolve()).replace('\\', '/')
    return '/' + value[0].lower() + value[2:] if len(value) > 1 and value[1] == ':' else value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bash', default='bash')
    args = p.parse_args()
    original_root = screen.ROOT
    (original_root / 'tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=original_root / 'tmp') as folder:
        root = Path(folder)
        screen.ROOT = root
        paths = root / 'paths.json'
        paths.write_text(json.dumps({'train': [], 'validation': {}}))
        class Options:
            group = 'ENGINEERING-MOEFS-QUEUE'
            paths_file = str(paths)
            seed = 10
            micro_batch = 16
            image_backend = 'npy_mmap'
            eval_every = 1000
        group, configs = screen.prepare(Options)
        a, b = [json.loads(configs[name].read_text()) for name in ('B0', 'A1-G4')]
        assert a.pop('moe_fs') is False and b.pop('moe_fs') is True and a == b
        assert a['record_updates'] == 1000 and a['total_updates'] == 1000000
        screen.prepare(Options)
        Options.micro_batch = 8
        try:
            screen.prepare(Options)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed frozen microbatch accepted')
        run_ids = {'B0': 'fixture-B0-s10', 'A1-G4': 'fixture-A1-G4-s10'}
        for label, run_id in run_ids.items():
            out = root / 'runs' / run_id
            out.mkdir(parents=True)
            rows = [dict(update=u, metrics={'DIV2K_valid': {'psnr': v, 'ssim': .9},
                                            'Set5': {'psnr': 99 if label == 'B0' else 1, 'ssim': .9}})
                    for u, v in ((5000, 30), (10000, 31), (15000, 32), (20000, 33))]
            if label == 'A1-G4':
                for row in rows:
                    row['metrics']['DIV2K_valid']['psnr'] += .1
            (out / 'epochs.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        screen.compare(group, run_ids, 20000)
        comparison = json.loads((group / 'comparison-s10-u20000.json').read_text())
        assert comparison['rows']['B0']['tail_updates'] == [10000, 15000, 20000]
        assert abs(comparison['validation_delta']['psnr'] - .1) < 1e-10
        # All executable children are mocks: no GPU work, no shutdown.
        fake = root / 'bin'
        fake.mkdir()
        (fake / 'flock').write_text('#!/usr/bin/env bash\nexit "${TEST_LOCK_CODE:-0}"\n')
        python = fake / 'fake_python'
        python.write_text('#!/usr/bin/env bash\nexit "${TEST_EXIT_CODE:-0}"\n')
        for executable in fake.iterdir():
            executable.chmod(0o755)
        for exit_code, lock_code in ((0, 0), (7, 0), (0, 1)):
            job = root / f'worker-{exit_code}-{lock_code}'
            job.mkdir()
            env = dict(os.environ, PYTHON_BIN=posix(python), TEST_EXIT_CODE=str(exit_code),
                       TEST_LOCK_CODE=str(lock_code))
            runner = root / 'runner.sh'
            runner.write_text(f'#!/usr/bin/env bash\nexport PATH="{posix(fake)}:$PATH"\n'
                              f'exec bash research/moe_fs/worker.sh "{posix(job)}"\n')
            result = subprocess.run([args.bash, posix(runner)], cwd=original_root, env=env)
            expected = 75 if lock_code else exit_code
            assert result.returncode == expected
            assert f'exit_code={expected}' in (job / 'status.txt').read_text()
    print('PASS immutable paired configs, fixed-tail validation comparison, worker exit/lock status.')


if __name__ == '__main__':
    main()
