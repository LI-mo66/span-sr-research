"""Attach to the existing Linux queue, back up at 20k, then release instance."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--server-pid', type=int, required=True)
    args = parser.parse_args()
    root = args.code_root.resolve()
    group = 'FEAT-SPAN-KD-OUTPUT-50K-01'
    label = 'SPAN-KD-Output'
    run_id = group + '-' + label + '-s10'
    run = root / 'runs' / run_id
    logs = root / 'logs/output_kd' / group
    records = Path('/home/featurize/work/span-x4-research/records')
    release = shutil.which('featurize')
    if not release:
        raise RuntimeError('featurize CLI missing; watcher not armed')
    sys.path.insert(0, str(root / 'research/output_kd'))
    import server
    import torch
    assert server.ROOT.resolve() == root

    def identity(pid, marker, token):
        proc = Path('/proc') / str(pid)
        cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if marker not in cmd or token not in cmd or (proc / 'cwd').resolve() != root:
            raise RuntimeError(f'Unexpected process {pid}; preserved')
        return (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]

    parent_start = identity(args.server_pid, 'output_kd/server.py', group)
    child_file = Path(f'/proc/{args.server_pid}/task/{args.server_pid}/children')
    matches = []
    for value in child_file.read_text().split():
        pid = int(value)
        try:
            start = identity(pid, 'output_kd/entry.py', run_id)
            matches.append((pid, start))
        except (OSError, RuntimeError):
            pass
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one matching training child; watcher not armed')
    trainer_pid, trainer_start = matches[0]

    def check_processes():
        if identity(args.server_pid, 'output_kd/server.py', group) != parent_start:
            raise RuntimeError('Queue PID was reused')
        if identity(trainer_pid, 'output_kd/entry.py', run_id) != trainer_start:
            raise RuntimeError('Trainer PID was reused')

    def rows(folder):
        result = []
        for line in (folder / 'epochs.jsonl').read_text().splitlines():
            try:
                row = json.loads(line)
                if 'update' in row:
                    result.append(row)
            except json.JSONDecodeError:
                pass  # A concurrent final append can be incomplete until next poll.
        return sorted(result, key=lambda row: row['update'])

    print(f'ARMED: queue={args.server_pid}, trainer={trainer_pid}, target epoch20/update20000', flush=True)
    previous = None
    while True:
        check_processes()
        history = rows(run) if (run / 'epochs.jsonl').exists() else []
        update = history[-1]['update'] if history else 0
        if update != previous:
            print(f'Last completed evaluation: {update}/20000', flush=True)
            previous = update
        if update > 20000:
            raise RuntimeError('Already past epoch20; no automatic release or data changes')
        if update == 20000:
            break
        time.sleep(.25)

    # Completed epoch record is appended only after latest/numbered/best saves.
    # Pause both processes to prevent further writes during persistent backup.
    check_processes()
    os.kill(args.server_pid, signal.SIGSTOP)
    os.kill(trainer_pid, signal.SIGSTOP)
    print('Paused at the completed 20k checkpoint. Any failure now leaves the instance paused and retained.', flush=True)
    state = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
    if state['update'] != 20000 or state['epoch'] != 20:
        raise RuntimeError('20k checkpoint mismatch; instance retained, processes remain paused')
    del state
    assert (run / 'epoch00020_u20000.pt').is_file()

    def selected(folder):
        history = [row for row in rows(folder) if row['update'] <= 20000]
        if not history or history[-1]['update'] != 20000:
            raise RuntimeError(f'Incomplete 20k history: {folder}')
        best = max(history, key=lambda row: row['metrics']['DIV2K_valid']['psnr'])
        weight = folder / f'epoch{best["epoch"]:05d}_u{best["update"]}.pt'
        if not weight.is_file():
            raise FileNotFoundError(weight)
        return dict(selected_update=best['update'], checkpoint=weight.name,
                    checkpoint_sha256=server.sha(weight), metrics=best['metrics'])

    baseline = records / 'FEAT-MOEFS-G4-LR2E4-20K-01-1791983354063378494/runs/FEAT-MOEFS-G4-LR2E4-20K-01-B0-s10'
    candidate, reference = selected(run), selected(baseline)
    result = dict(stop_update=20000, stop_epoch=20, completed_requested_budget=True,
                  original_configured_cap=50000, B0_20k=reference, candidate_20k=candidate,
                  note='External requested cap; frozen training configuration unchanged; compare only <=20k histories')
    result['delta_vs_B0_20k'] = {
        name: {metric: values[metric]-reference['metrics'][name][metric] for metric in ('psnr','ssim')}
        for name, values in candidate['metrics'].items()}
    (logs / 'comparison-20000.json').write_text(json.dumps(result, indent=2))
    # Ignore any uncheckpointed work begun between epoch append and SIGSTOP.
    # Original live logs are retained verbatim, not silently truncated.
    destination = server.backup(SimpleNamespace(group=group, variants=[label], backup_root=records), logs, 0)
    if not (destination / 'BACKUP_COMPLETE').is_file():
        raise RuntimeError('Backup incomplete; instance retained, processes remain paused')
    status_path = records / f'{group}-stop20-status.json'
    status = dict(exit_code=0, completed_updates=20000, backup_exit_code=0,
                  backup=str(destination), release_requested=False)
    status_path.write_text(json.dumps(status, indent=2))
    os.sync()
    check_processes()
    # SIGKILL also terminates a stopped process; only the verified job is targeted.
    os.kill(trainer_pid, signal.SIGKILL)
    os.kill(args.server_pid, signal.SIGKILL)
    status['release_requested'] = True
    status_path.write_text(json.dumps(status, indent=2))
    os.sync()
    print(f'20k backup verified: {destination}. Releasing instance now.', flush=True)
    released = subprocess.run([release, 'instance', 'release'], stdin=subprocess.DEVNULL)
    status['release_command_exit_code'] = released.returncode
    status_path.write_text(json.dumps(status, indent=2))
    os.sync()
    if released.returncode:
        raise RuntimeError('Release command failed; training stopped and backed up, release manually')


if __name__ == '__main__':
    main()
