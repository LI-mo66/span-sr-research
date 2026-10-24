#!/usr/bin/env bash
# Linux/Featurize only. No shutdown or instance-release capability.
set -euo pipefail
cd "$(dirname "$0")/../.."

group=${1:-FEAT-SG6-FT-10K-01}
data=${2:-/home/featurize/datasets}
reference=${3:-/home/featurize/work/span-x4-research/records/FEAT-LG6-SG6-50K-01-1792264356078917269}
records=${BACKUP_ROOT:-/home/featurize/work/span-x4-research/records}
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
[[ $(uname -s) == Linux ]] || { echo 'Linux server required'; exit 2; }
command -v nohup >/dev/null
command -v flock >/dev/null
[[ -d "$data" ]] || { echo "Missing dataset root: $data"; exit 2; }
[[ -f "$reference/BACKUP_COMPLETE" ]] || { echo "Missing completed SG6 backup: $reference"; exit 2; }
[[ "$records" == /home/featurize/work/* ]] || { echo 'Backups must be under persistent work'; exit 2; }
[[ -z $(git status --porcelain) ]] || { git status --short; exit 2; }

"$python_bin" - <<'PY'
import torch, numpy, cv2, scipy, matplotlib
print('torch:', torch.__version__, 'numpy:', numpy.__version__, 'cv2:', cv2.__version__,
      'scipy:', scipy.__version__, 'matplotlib:', matplotlib.__version__)
assert torch.cuda.is_available(), 'CUDA unavailable; use a GPU server'
print('GPU:', torch.cuda.get_device_name())
assert torch.__version__ == '2.2.2+cu121', 'PyTorch differs from SG6'
assert torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 4090', 'GPU differs from SG6'
assert numpy.__version__ == '1.26.4'
assert cv2.__version__ == '4.10.0'
assert scipy.__version__ == '1.13.1'
assert matplotlib.__version__ == '3.8.4'
assert (torch.ones(4, device='cuda')*2).sum().item() == 8
PY

"$python_bin" - "$data" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
link, source = root/'Flickr2K', root/'downloads/Flickr2K'
extensions = {'.png','.bmp','.jpg','.jpeg'}
def complete(folder):
    counts = []
    for name in ('Flickr2K_HR','Flickr2K_LR_bicubic/X4'):
        path = folder/name
        n = sum(p.is_file() and p.suffix.lower() in extensions for p in path.iterdir()) if path.is_dir() else 0
        print(path, f'{n}/2650'); counts.append(n)
    return counts == [2650,2650]
if not complete(link):
    if not complete(source): raise SystemExit('Flickr2K incomplete; training not started')
    if link.is_symlink(): link.unlink()
    elif link.exists(): raise SystemExit('Flickr2K is a real directory; preserved')
    link.symlink_to(source, target_is_directory=True)
    print('Restored Flickr2K link')
PY

mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>/home/featurize/span-x4-runtime/logs/server-gpu.lock
if ! flock -n 9; then echo 'GPU queue lock held; not started'; exit 75; fi
"$python_bin" - "$data" <<'PY'
import sys, subprocess
from pathlib import Path
runtime = Path('/home/featurize/span-x4-runtime')
targets = dict(datasets=Path(sys.argv[1]).resolve(), runs=runtime/'runs', logs=runtime/'logs')
for name,target in targets.items():
    path=Path(name)
    if path.is_symlink():
        if path.resolve()!=target.resolve(): raise RuntimeError(f'Unknown existing link: {path}')
    elif path.exists(): raise RuntimeError(f'Existing real directory/file preserved: {path}')
for name,target in targets.items():
    target.mkdir(parents=True,exist_ok=True)
    if not Path(name).is_symlink(): Path(name).symlink_to(target,target_is_directory=True)
    print(name,'->',target)
exclude=Path(subprocess.check_output(['git','rev-parse','--git-path','info/exclude'],text=True).strip())
text=exclude.read_text() if exclude.exists() else ''
for name in targets:
    rule='/'+name
    if rule not in text.splitlines(): text+='\n'+rule+'\n'
exclude.parent.mkdir(parents=True,exist_ok=True);exclude.write_text(text)
PY

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
"$python_bin" research/sg6_finetune/verify.py --device cuda
[[ -z $(git status --porcelain) ]] || { git status --short; exit 2; }
job="logs/sg6_finetune/$group/job-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$job" "$records"
flock -u 9
exec 9>&-
nohup "$python_bin" -u research/sg6_finetune/ft_queue.py \
    --group "$group" --data-root "$data" --backup "$reference" \
    --records "$records" --job "$job" \
    > "$job/nohup.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$job/pid"
printf 'PID=%s\nLog=%s/%s/nohup.log\nStatus=%s/%s/status.json\n' "$pid" "$PWD" "$job" "$PWD" "$job"
echo 'Only A-L1 and B-MSE 10K fine-tuning are scheduled. EMA and checkpoint averaging included. Automatic shutdown is disabled.'
