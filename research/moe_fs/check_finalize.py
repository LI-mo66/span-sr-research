"""Synthetic plot/backup test; no platform release or real training."""
import json
import os
from pathlib import Path
from finalize import backup, sha256

ROOT = Path(__file__).resolve().parents[2]
fixture = ROOT / 'tmp' / 'moe-fs-finalize-check'
os.environ.setdefault('MPLCONFIGDIR', str(fixture / 'matplotlib-cache'))
group = 'PLOT-BACKUP-ENGINEERING'
source = fixture / 'logs/moe_fs' / group
source.mkdir(parents=True, exist_ok=True)
for label, enabled in (('B0', False), ('A1-G4', True)):
    (source / f'{label}-s10.json').write_text(json.dumps(dict(moe_fs=enabled, seed=10)))
    folder = fixture / 'runs' / f'{group}-{label}-s10'
    folder.mkdir(parents=True, exist_ok=True)
    rows = [dict(update=u, train_l1_mean=.02 / (i+1), metrics={name: dict(psnr=25+i+(0.1 if enabled else 0), ssim=.8+i*.01)
            for name in ('DIV2K_valid', 'Set5', 'Set14', 'B100', 'Urban100', 'Manga109')})
            for i, u in enumerate((1000, 2000, 3000))]
    (folder / 'epochs.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    (folder / 'loss_steps.jsonl').write_text('\n'.join(json.dumps(dict(update=u, train_l1=.02)) for u in range(1, 3001))+'\n')
    (folder / 'latest.pt').write_bytes(b'engineering mock checkpoint, not a model')
output = backup(fixture, group, fixture / 'durable', success=True)
assert (output / 'BACKUP_COMPLETE').exists()
manifest = json.loads((output / 'backup-manifest.json').read_text())
assert all(sha256(output / row['path']) == row['sha256'] for row in manifest['files'])
assert len(list((output / 'curves').rglob('*.png'))) == 23
print('PASS verified backup and 23 PNG/SVG charts; fixture:', output)
