"""Protocol/selection regression tests: synthetic files, never start B0 training."""
import copy
import hashlib
import importlib.util
import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class Checks(unittest.TestCase):
    def test_frozen_protocol_and_baseline_refusal(self):
        run = load('a6_run_contract', 'run.py')
        cfg = json.loads((HERE/'configs/A6-lr2e4.json').read_text())
        for bad in (dict(cfg, a6=False), dict(cfg, lr=5e-4), dict(cfg, micro_batch=4),
                    dict(cfg, total_updates=50000), dict(cfg, self_ensemble=True),
                    dict(cfg, easa=True), dict(cfg, record_updates=5000)):
            with self.assertRaises(ValueError): run.validate_config(bad)
        # Dedicated engineering fixtures keep A6, but can use tiny synthetic data.
        run.validate_config(dict(cfg, engineering_only=True))
        original = run.argparse.ArgumentParser.add_argument
        def shared_main():
            p = run.argparse.ArgumentParser(); p.add_argument('--screen-updates', type=int, choices=[5000])
            self.assertEqual(p.parse_args(['--screen-updates','50000']).screen_updates, 50000)
        with patch.object(run.a6_runtime.trainer, 'main', shared_main): run.train_main()
        self.assertIs(run.argparse.ArgumentParser.add_argument, original)

    def test_path_reconstruction(self):
        prepare = load('a6_prepare_contract', 'prepare_reuse.py')
        paths = prepare.reconstructed_paths(dict(train=[dict(hr='/a/HR/1.png', lr='/a/LR/1x4.png'),
                                                        dict(hr='/b/HR/2.png',lr='/b/LR/2.png')],
                                                   DIV2K_valid=[dict(hr='/v/HR/1.png', lr='/v/LR/1x4.png')]))
        self.assertEqual([r['lr_suffix'] for r in paths['train']], ['x4',''])

    def test_comparison_selection_curves_and_corruption(self):
        compare = load('a6_compare_contract', 'compare.py')
        root = HERE.parents[1]/'tmp'/('a6-compare-'+uuid.uuid4().hex[:8]); root.mkdir(parents=True)
        base, candidate = root/'B0', root/'A6'; base.mkdir(); candidate.mkdir()
        def rows(variant):
            return [dict(epoch=i, update=i*1000, metrics=dict(
                DIV2K_valid=dict(psnr=(30. if i == 48 else 29.) if not variant else (31. if i in (47,48) else 29.), ssim=.8),
                Urban100=dict(psnr=(25. if not variant else (27. if i == 49 else 26.)), ssim=.7))) for i in range(1,51)]
        a, b = rows(False), rows(True)
        for path, history in ((base,a), (candidate,b)):
            (path/'epochs.jsonl').write_text('\n'.join(json.dumps(r) for r in history))
        data, cfg, env = {}, dict(a6=True), dict(torch='synthetic')
        audit = dict(baseline_run=str(base), candidate_source='src', candidate_config=cfg,
                     environment=env, baseline_selected=a[47], baseline_history=a,
                     data_fingerprint=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest())
        (root/'audit.json').write_text(json.dumps(audit))
        (candidate/'manifest.json').write_text(json.dumps(dict(commit='src',config=cfg,environment=env,data=data)))
        (candidate/'epoch00047_u47000.pt').touch()
        per_image = [dict(image='a.png', psnr=30., ssim=.8)]
        (base/'DIV2K_valid_epoch00048_u48000.json').write_text(json.dumps(per_image))
        per_image[0]['psnr']=31.
        (candidate/'DIV2K_valid_epoch00047_u47000.json').write_text(json.dumps(per_image))
        args = ['compare','--audit',str(root/'audit.json'),'--candidate-run',str(candidate),'--output',str(root/'comparison.json')]
        with patch.object(sys,'argv',args): compare.main()
        result=json.loads((root/'comparison.json').read_text())
        self.assertEqual(result['candidate_checkpoint'],'epoch00047_u47000.pt')
        self.assertEqual(result['validation_selected']['Urban100']['A6']['psnr'],26.)
        self.assertEqual(result['fixed_50000']['DIV2K_valid']['A6']['psnr'],29.)
        self.assertAlmostEqual(result['last3_evaluation_mean']['DIV2K_valid']['delta']['psnr'],1/3)
        self.assertEqual(result['paired_validation_selected']['DIV2K_valid']['win_rate'],1.)
        self.assertTrue((root/'comparison.svg').exists() and (root/'comparison.csv').exists())
        self.assertIn('共享', (root/'comparison.md').read_text(encoding='utf8'))
        (candidate/'epochs.jsonl').write_text('\n'.join(json.dumps(r) for r in b[:-1]))
        with patch.object(sys,'argv',args), self.assertRaises(ValueError): compare.main()
        # Backup refuses incomplete runs, never modifies archived B0.
        backup = load('a6_backup_contract','backup.py')
        group=root/'logs/a6/group'; group.mkdir(parents=True)
        (group/'comparison-50000.json').write_text(json.dumps(dict(candidate_checkpoint='epoch00047_u47000.pt')))
        run=root/'runs/A6-s10'; run.mkdir(parents=True)
        with self.assertRaises(FileNotFoundError): backup.backup(root,group,'A6-s10',root/'records')
        for name in ('manifest.json','epochs.jsonl','loss_steps.jsonl','latest.pt',
                     'best_validation.pt','epoch00047_u47000.pt'):
            (run/name).write_text('synthetic backup fixture')
        output = backup.backup(root,group,'A6-s10',root/'records')
        self.assertTrue((output/'BACKUP_COMPLETE').is_file())
        inventory=json.loads((output/'backup-manifest.json').read_text())
        for entry in inventory['files']:
            self.assertEqual(backup.digest(output/entry['path']),entry['sha256'])
        real_copy=backup.shutil.copy2
        def corrupt(source,target):
            real_copy(source,target); Path(target).write_text('corrupt')
        with patch.object(backup.shutil,'copy2',corrupt), self.assertRaises(IOError):
            backup.backup(root,group,'A6-s10',root/'corrupt-records')
        self.assertFalse(list((root/'corrupt-records').glob('*/BACKUP_COMPLETE')))


if __name__ == '__main__': unittest.main()
