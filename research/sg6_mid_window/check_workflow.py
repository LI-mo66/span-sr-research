"""Meaningful workflow checks: cap, checkpoint selection, paired data, backup corruption."""
import argparse
import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
import run
from prepare import history, paths_from_manifest
from compare import compare_one, paired
from finalize import backup, digest
from monitor import attach
from adapter import PartialInteraction

ROOT=Path(__file__).resolve().parents[2]


def temporary():
    root=ROOT/'tmp'/('mid-test-'+uuid.uuid4().hex[:8]);root.mkdir(parents=True)
    return root


class Checks(unittest.TestCase):
    def test_cap_and_frozen_settings(self):
        def shared():
            p=argparse.ArgumentParser();p.add_argument('--screen-updates',type=int,choices=[5000,20000,100000])
            self.assertEqual(p.parse_args(['--screen-updates','50000']).screen_updates,50000)
        original=argparse.ArgumentParser.add_argument
        with patch.object(run.mid_runtime.trainer,'main',shared):run.train_main()
        self.assertIs(argparse.ArgumentParser.add_argument,original)
        for kind in ('A','C'):
            cfg=json.loads((Path(__file__).parent/'configs'/f'{kind}.json').read_text())
            run.validate_config(cfg)
            with self.assertRaises(ValueError):run.validate_config(dict(cfg,total_updates=50000))
            with self.assertRaises(ValueError):run.validate_config(dict(cfg,seed=11))

    def test_paths_preserve_source_order(self):
        data=dict(train=[dict(hr='/a/HR/1.png',lr='/a/LR/1x4.png'),dict(hr='/b/HR/2.png',lr='/b/LR/2.png')])
        self.assertEqual([r['lr_suffix'] for r in paths_from_manifest(data)['train']],['x4',''])

    def test_one_div_selected_checkpoint_earliest_tie(self):
        root=temporary();metrics=dict(DIV2K_valid=dict(psnr=30.,ssim=.8),B100=dict(psnr=27.,ssim=.7))
        cfg={'synthetic':True}
        audit=dict(candidate_source='commit',candidate_configs={'A':cfg},fixed_base='base',
                   reference_weight='epoch00048_u48000.pt',reference_selected=dict(metrics=metrics),reference_final=dict(metrics=metrics))
        (root/'manifest.json').write_text(json.dumps(dict(commit='commit',config=cfg)))
        rows=[dict(epoch=i,update=i*1000,metrics=dict(DIV2K_valid=dict(psnr=31. if i in (47,48) else 29.,ssim=.81),
                  B100=dict(psnr=28. if i==49 else 27.1,ssim=.71))) for i in range(1,51)]
        (root/'epochs.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        (root/'epoch00047_u47000.pt').touch()
        result,_=compare_one(audit,root,'A',include_paired=False)
        self.assertEqual(result['candidate_checkpoint'],'epoch00047_u47000.pt')
        self.assertEqual(result['validation_selected']['B100']['SG6_A']['psnr'],27.1)
        (root/'epochs.jsonl').write_text('\n'.join(json.dumps(r) for r in rows[:-1]))
        with self.assertRaises(ValueError):history(root)

    def test_paired_images_and_source_means(self):
        root=temporary();old=root/'old';new=root/'new';old.mkdir();new.mkdir()
        a=[dict(image='1.png',psnr=20.,ssim=.7),dict(image='2.png',psnr=30.,ssim=.9)]
        b=[dict(image='2.png',psnr=29.,ssim=.9),dict(image='1.png',psnr=22.,ssim=.7)]
        for folder,rows in ((old,a),(new,b)):
            (folder/'B100_epoch00048_u48000.json').write_text(json.dumps(rows))
        before=dict(epoch=48,update=48000,metrics={'B100':dict(psnr=25.,ssim=.8)})
        after=dict(epoch=48,update=48000,metrics={'B100':dict(psnr=25.5,ssim=.8)})
        result=paired(old,new,before,after)['B100']
        self.assertEqual((result['improved'],result['worse']),(1,1))
        self.assertEqual(result['mean_delta_psnr'],.5)
        b[0]['image']='unexpected.png'
        (new/'B100_epoch00048_u48000.json').write_text(json.dumps(b))
        with self.assertRaises(ValueError):paired(old,new,before,after)

    def test_backup_both_runs_and_corruption(self):
        root=temporary();group=root/'logs/sg6_mid_window/GROUP';group.mkdir(parents=True)
        (group/'comparison-AC-50000.json').write_text('{}')
        for kind in ('A','C'):
            (group/f'comparison-{kind}-50000.json').write_text(json.dumps(dict(candidate_checkpoint='epoch00048_u48000.pt')))
            folder=root/'runs'/f'GROUP-{kind}-s10';folder.mkdir(parents=True)
            for name in ('manifest.json','epochs.jsonl','loss_steps.jsonl','latest.pt','best_validation.pt','epoch00048_u48000.pt'):
                (folder/name).write_text('test fixture')
        output=backup(root,group,root/'records')
        self.assertTrue((output/'BACKUP_COMPLETE').is_file())
        for row in json.loads((output/'backup-manifest.json').read_text())['files']:
            self.assertEqual(digest(output/row['path']),row['sha256'])
        from finalize import shutil
        original=shutil.copy2
        def corrupt(source,target):original(source,target);Path(target).write_text('corrupt')
        with patch.object(shutil,'copy2',corrupt),self.assertRaises(IOError):backup(root,group,root/'corrupt')
        self.assertFalse(list((root/'corrupt').glob('*/BACKUP_COMPLETE')))

    def test_monitor_observational_only(self):
        torch.set_num_threads(2)
        class Mini(nn.Module):
            def __init__(self):
                super().__init__();self.block_3=nn.Module();self.block_3.interaction=PartialInteraction('C')
            def forward(self,x):return self.block_3.interaction(x)
        torch.manual_seed(10);original=Mini();watched=Mini();watched.load_state_dict(original.state_dict())
        output=temporary()/'monitor.jsonl'
        attach(watched,dict(effective_batch=2,micro_batch=1,mid_interaction='C'),output)
        x=torch.randn(1,52,3,5);target=torch.randn_like(x)
        optimizers=[torch.optim.Adam(m.parameters(),lr=2e-4) for m in (original,watched)]
        rng=torch.get_rng_state().clone()
        for model,opt in zip((original,watched),optimizers):
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                for _ in range(2):((model(x)-target).square().mean()/2).backward()
                opt.step()
            model.eval();model(x)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,watched.state_dict()[k]) for k,v in original.state_dict().items()))
        rows=[json.loads(s) for s in output.read_text().splitlines()]
        self.assertEqual([r['update'] for r in rows],[1])


if __name__=='__main__':unittest.main()
