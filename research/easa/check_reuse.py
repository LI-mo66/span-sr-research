"""Verify cap-only CLI extension, archived path recovery and checkpoint selection."""
import importlib.util
import json
import sys
import unittest
import uuid
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReuseChecks(unittest.TestCase):
    def test_cap_is_execution_only(self):
        run = load('easa_run_check', 'run.py')
        original = run.argparse.ArgumentParser.add_argument
        def shared_main():
            parser = run.argparse.ArgumentParser()
            parser.add_argument('--screen-updates', type=int, choices=[5000, 20000, 100000])
            self.assertEqual(parser.parse_args(['--screen-updates', '50000']).screen_updates, 50000)
        with patch.object(run.easa_runtime.trainer, 'main', shared_main):
            run.train_main()
        self.assertIs(run.argparse.ArgumentParser.add_argument, original)
        cfg = json.loads((HERE / 'configs/E1-mid-lr2e4.json').read_text())
        cfg['micro_batch'] = 16
        run.validate_config(cfg)
        self.assertEqual(cfg['total_updates'], 1000000)
        self.assertEqual(cfg['milestones'], [200000, 400000, 600000, 800000])
        with self.assertRaises(ValueError):
            run.validate_config(dict(cfg, total_updates=50000))

    def test_paths_recover_group_order_and_suffix(self):
        prepare = load('easa_prepare_check', 'prepare_reuse.py')
        data = dict(train=[dict(hr='/a/HR/01.png', lr='/a/LR/01x4.png'),
                           dict(hr='/b/HR/02.png', lr='/b/LR/02.png')],
                    DIV2K_valid=[dict(hr='/v/HR/03.png', lr='/v/LR/03x4.png')])
        paths = prepare.reconstructed_paths(data)
        self.assertEqual([p['hr'] for p in paths['train']], ['/a/HR', '/b/HR'])
        self.assertEqual([p['lr_suffix'] for p in paths['train']], ['x4', ''])
        self.assertEqual(paths['validation']['DIV2K_valid']['lr_suffix'], 'x4')
        with self.assertRaises(ValueError):
            prepare.reconstructed_paths(dict(train=[dict(hr='/a/01.png', lr='/a/different.png')]))

    def test_one_validation_selected_checkpoint_and_fixed_final(self):
        compare = load('easa_compare_check', 'compare_reuse.py')
        temp_root = HERE.parents[1] / 'tmp'
        temp_root.mkdir(exist_ok=True)
        # Keep fixtures under ignored workspace tmp/; Windows sandbox does not
        # support tempfile's restrictive ACL/chmod cleanup behavior.
        with nullcontext(temp_root / ('reuse-test-' + uuid.uuid4().hex[:10])) as folder:
            root = Path(folder)
            assert root.resolve().is_relative_to(temp_root.resolve())
            root.mkdir()
            metrics = dict(DIV2K_valid=dict(psnr=30., ssim=.8),
                           Urban100=dict(psnr=25., ssim=.7))
            audit = dict(candidate_source='source', baseline_record='b546036',
                         baseline_selected=dict(metrics=metrics), baseline_final=dict(metrics=metrics))
            (root / 'audit.json').write_text(json.dumps(audit))
            (root / 'manifest.json').write_text(json.dumps(dict(commit='source')))
            history = []
            for epoch in range(1, 51):
                history.append(dict(epoch=epoch, update=epoch*1000,
                    metrics=dict(DIV2K_valid=dict(psnr=31. if epoch in (47, 48) else 29., ssim=.81),
                                 Urban100=dict(psnr=27. if epoch == 49 else 26., ssim=.72))))
            (root / 'epochs.jsonl').write_text('\n'.join(json.dumps(r) for r in history))
            (root / 'epoch00047_u47000.pt').touch()
            args = ['compare', '--audit', str(root / 'audit.json'), '--candidate-run', str(root),
                    '--output', str(root / 'comparison.json')]
            with patch.object(sys, 'argv', args):
                compare.main()
            result = json.loads((root / 'comparison.json').read_text())
            self.assertEqual(result['candidate_checkpoint'], 'epoch00047_u47000.pt')
            self.assertEqual(result['validation_selected']['Urban100']['EASA']['psnr'], 26.)
            self.assertEqual(result['fixed_50000']['DIV2K_valid']['EASA']['psnr'], 29.)
            (root / 'epochs.jsonl').write_text('\n'.join(json.dumps(r) for r in history[:-1]))
            with patch.object(sys, 'argv', args), self.assertRaises(ValueError):
                compare.main()


if __name__ == '__main__':
    unittest.main()
