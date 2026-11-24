"""Validate archived transcripts/metrics/hashes with standard-library Python."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
report = json.loads((HERE / 'results.json').read_text(encoding='utf8'))
variants = report['weight_fusion']
datasets = ('DIV2K_valid', 'Set5', 'Set14', 'B100', 'Urban100', 'Manga109')
targets = datasets[1:]
assert len(variants) == 7 and len(report['patch64']) == 4 and len(report['patch96']) == 2
raw = (HERE / 'evidence/weight-fusion-evaluation.txt').read_text(encoding='utf8')
for mode, number, heading in (
    ('x8', 7, '所有版本同一权重的×8指标'), ('plain', 9, '普通推理指标'),
):
    match = re.search(re.escape(f'========== {number}. {heading} ==========')
                      + r'\s*\n(.*?)(?=\n========== |\Z)', raw, re.S)
    assert match
    parts = re.split(r'\n版本：\s*(\S+)\s*\n', '\n' + match[1])
    for i in range(1, len(parts), 2):
        for line in parts[i+1].splitlines():
            fields = line.split()
            if fields and fields[0] in datasets:
                key = fields[0] + ('_x8' if mode == 'x8' else '')
                metric = variants[parts[i]]['metrics'][key]
                assert float(fields[1]) == metric['psnr']
                assert float(fields[2]) == metric['ssim']
for name, row in variants.items():
    assert len(row['metrics']) == 12
    assert all(math.isfinite(x) for m in row['metrics'].values() for x in m.values())
    deltas = [row['metrics'][n+'_x8']['psnr']
              - variants['original_982k']['metrics'][n+'_x8']['psnr'] for n in targets]
    assert abs(sum(deltas)/5 - row['mean_five_x8_delta']) < 1e-12
    assert min(deltas) == row['min_five_x8_delta']
    assert all(x > 0 for x in deltas) == row['all_five_x8_improved']
for arm, label in (('l1_lr5', 'l1_4k'), ('mse_lr5', 'mse_7k')):
    for name, score in report['patch96'][arm]['selected_x8'].items():
        metric = variants[label]['metrics'][name+'_x8']
        assert score['psnr'] == metric['psnr'] and score['ssim'] == metric['ssim']
with (HERE / 'fusion-metrics.csv').open(encoding='utf8', newline='') as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 84
for row in rows:
    key = row['dataset'] + ('_x8' if row['mode'] == 'x8' else '')
    score = variants[row['variant']]['metrics'][key]
    assert float(row['psnr_db']) == score['psnr'] and float(row['ssim']) == score['ssim']
with (HERE / 'patch96-metrics.csv').open(encoding='utf8', newline='') as stream:
    patch_rows = list(csv.DictReader(stream))
assert len(patch_rows) == 36
for row in patch_rows:
    score = report['patch96'][row['arm']][row['selection']][row['dataset']]
    assert float(row['x8_psnr_db']) == score['psnr']
    assert float(row['x8_ssim']) == score['ssim']
    assert float(row['reported_delta_psnr']) == score['reported_delta_psnr']
for shape in ('180x320', '360x640'):
    data = json.loads((HERE / f'efficiency/f0-x2-profile-{shape}-20261123.json').read_text())
    assert sum(op['flops'] for op in data['operations']) == data['flops']
    assert data['deployed_parameters'] == 552003
manifest = HERE / 'archive-manifest.json'
if manifest.is_file():
    for item in json.loads(manifest.read_text())['files']:
        path = HERE / item['path']
        assert path.is_file() and path.stat().st_size == item['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path
print('PASS: raw fusion tables,7x12 metric groups, selected patch96 references,84 CSV rows, FLOPs sums and archive file hashes')
