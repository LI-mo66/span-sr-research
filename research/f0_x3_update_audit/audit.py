"""Abort-only native replay. No training-code rewriting or optimization rescue."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'research/multiscale_f0'), str(ROOT/'research')]
import fusion
import trainer
from image_io import enable_cache

SOURCE_SHA = 'b89b03032267db41405124807a53d2ac28aea241e601919049cf917cc558563b'


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore(value):
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch']); torch.cuda.set_rng_state_all(value['cuda'])


class RandomTrace:
    def __init__(self, source):
        self.source, self.draws = source, []

    def randint(self, a, b):
        value = self.source.randint(a, b)
        self.draws.append(dict(op='randint', a=a, b=b, value=value))
        return value

    def random(self):
        value = self.source.random()
        self.draws.append(dict(op='random', value=value))
        return value


def norm(parameters):
    return torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(p.grad.detach(), 2) for p in parameters if p.grad is not None
    ]), 2)


def probe(model, batch):
    # Keep training-mode Conv3XC execution. No BN/dropout is allowed by preflight.
    with torch.no_grad():
        x, y = batch
        value = model(x)
        return float(torch.nn.functional.l1_loss(value, y))


def step(model, optimizer, scheduler, batches, clip=.1, accum=4):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for x, y in batches:
        prediction = model(x)
        assert prediction.shape == y.shape
        loss = torch.nn.functional.l1_loss(prediction, y) / accum
        raw = float(loss.detach()) * accum
        if not math.isfinite(raw) or raw > 1.:
            return dict(original_loss_guard=True, micro_losses=losses + [raw if math.isfinite(raw) else str(raw)])
        loss.backward(); losses.append(raw)
    assert len(losses)==accum
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    before_parameters = {n: p.detach().clone() for n, p in named}
    before_grad = float(torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True))
    after_grad = float(norm(parameters))
    assert after_grad <= clip + 5e-6, ('Post-clip norm violation', after_grad)
    gradients = {n: p.grad.detach().clone() for n, p in named}
    optimizer.step(); scheduler.step()
    changes = []
    for name, p in named:
        assert torch.isfinite(p).all(), name
        old = before_parameters[name]
        delta = p.detach() - old
        size, pn = float(delta.norm()), float(old.norm())
        changes.append(dict(name=name, update_norm=size, parameter_norm=pn,
                            relative_update=size/max(pn, 1e-12),
                            update_absmax=float(delta.abs().max())))
    return dict(mean_l1=sum(losses)/len(losses), micro_losses=losses,
                gradient_norm_before_clip=before_grad, gradient_norm_after_clip=after_grad,
                clip_applied=before_grad > clip, gradients_after_clip=gradients,
                changes=changes)


def stats(t):
    t = t.detach()
    maximum=float(t.abs().max())
    rms=float(t.double().square().mean().sqrt())
    return dict(shape=list(t.shape), absmax=maximum if math.isfinite(maximum) else str(maximum),
                rms=rms if math.isfinite(rms) else str(rms),
                finite_fraction=float(torch.isfinite(t).float().mean()),
                above8_fraction=float((t.abs()>8).float().mean()))


class FeatureTrace:
    def __init__(self, model):
        self.enabled, self.rows, self.handles = False, [], []
        for name in ('conv_1', 'conv_2', 'conv_cat', 'upsampler.0', 'image_residual'):
            def hook(module, inputs, output, label=name):
                if self.enabled:
                    with torch.no_grad(): self.rows.append(dict(layer=label, output=stats(output)))
            self.handles.append(model.get_submodule(name).register_forward_hook(hook))
        for i in range(1, 7):
            wrapper = getattr(model, f'block_{i}')
            block = getattr(wrapper, 'original', wrapper)
            cache = {}
            def third_hook(module, inputs, output, holder=cache):
                if self.enabled: holder['third'] = output.detach()
            self.handles.append(block.c3_r.register_forward_hook(third_hook))
            def main_hook(module, inputs, outputs, holder=cache, label=f'block_{i}'):
                if self.enabled:
                    with torch.no_grad():
                        raw = (holder.pop('third') + inputs[0].detach()) * outputs[2].detach()
                        self.rows.append(dict(layer=label, main=stats(outputs[0]),
                            first_tap=stats(outputs[1]), gate=stats(outputs[2]), raw_product=stats(raw)))
            self.handles.append(block.register_forward_hook(main_hook))


def snapshot(model, optimizer, scheduler, update, random_state):
    return dict(format='X3-native-update-diagnostic-v1', update=update,
                model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                scheduler=copy.deepcopy(scheduler.state_dict()), rng=random_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source_project.resolve(), args.output.resolve()
    checkpoint = source/'runs/F0-X3-CLIP01-U300K-14/epoch00295_u295000.pt'
    assert not (out/'protocol.json').exists(), 'Output already used'
    assert out.is_relative_to(Path('/home/featurize/work').resolve())
    assert torch.__version__ == '2.2.2+cu121', torch.__version__
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
    assert trainer.digest(checkpoint) == SOURCE_SHA
    metadata = json.loads((checkpoint.parent/'manifest.json').read_text())
    cfg = metadata['config']
    assert cfg['arm']=='F0' and cfg['model']['upscale']==3 and cfg['hr_patch']==192
    assert cfg['micro_batch']==16 and cfg['effective_batch']==64 and cfg['loader_workers']==0
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['update']==295000 and saved['epoch']==295
    assert saved['fingerprint']==metadata['fingerprint']
    assert [g['lr'] for g in saved['optimizer']['param_groups']]==[1e-4]
    assert saved['scheduler']['last_epoch']==295000
    assert saved['scheduler']['base_lrs']==[2e-4]
    assert saved['scheduler']['_last_lr']==[1e-4]
    assert dict(saved['scheduler']['milestones'])=={200000:1,400000:1,600000:1,800000:1}
    model_hashes = {}
    for name in ('fusion.py','sg_model.py','bound.py','gb_model.py','lg_model.py',
                 'attention.py','lksa.py','window_adapter.py','ensemble.py'):
        relative = 'research/multiscale_f0/'+name
        original = subprocess.check_output(['git','show','0748bd60c1eaddda4e0db7f01ba768a0f2a22be3:'+relative],cwd=ROOT)
        assert original == (ROOT/relative).read_bytes(), relative
        model_hashes[relative]=hashlib.sha256(original).hexdigest()
    paths = json.loads(Path(cfg['paths_file']).read_text(encoding='utf-8-sig'))
    trainer.SCALE = 3
    train = trainer.pairs(paths['train'])
    assert len(train)==len(metadata['data']['train'])==3450
    for index, ((hr, lr), row) in enumerate(zip(train, metadata['data']['train']), 1):
        assert str(hr)==row['hr'] and str(lr)==row['lr']
        assert trainer.digest(hr)==row['hr_sha256'] and trainer.digest(lr)==row['lr_sha256']
        high, low = trainer.aligned(hr, lr)
        assert min(low.shape[:2])>=64
        if index%500==0: print('Auditing ordered train data', index, '/3450', flush=True)
    enable_cache({'train':metadata['data']['train']}, args.cache)
    references = {}
    with (checkpoint.parent/'loss_steps.jsonl').open() as stream:
        for line in stream:
            if not line.strip(): continue
            row = json.loads(line)
            if 294001<=row['update']<=295700: references[row['update']] = row
    assert all(u in references for u in range(294001,295701)), 'Recent original loss history incomplete'
    assert all(math.isfinite(r['train_l1']) for r in references.values())
    history = [references[u]['train_l1'] for u in range(294001,295001)]
    assert max(1.,20*statistics.median(history))==1.
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True; torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    arch, _ = fusion.components()
    model = fusion.build_model(arch, cfg).cuda().train()
    model.load_state_dict(saved['model'], strict=True)
    assert not any(isinstance(m, (torch.nn.modules.batchnorm._BatchNorm,
                    torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)) for m in model.modules())
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg['lr'],
        betas=tuple(cfg['adam_betas']), eps=cfg['adam_eps'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,cfg['milestones'],gamma=cfg['gamma'])
    optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
    assert len(optimizer.state)==272
    # Independent probe sampling happens BEFORE restoring the original random stream.
    private = random.Random(12345)
    probe_samples = [trainer.sample(private.choice(train),192,private) for _ in range(4)]
    fixed_probe = tuple(torch.stack([s[k] for s in probe_samples]).cuda() for k in (0,1))
    source_rng = dict(python=saved['python_rng'],numpy=saved['numpy_rng'],
                      torch=saved['torch_rng'],cuda=saved['cuda_rng'])
    restore(source_rng)
    del saved
    feature_trace = FeatureTrace(model)
    protocol = dict(source_checkpoint=str(checkpoint), source_sha256=SOURCE_SHA,
        start=295000, maximum_stop=295700, lr=1e-4, clip=.1, model_unchanged=True,
        loss_history_match_atol=1e-7, source_fingerprint=metadata['fingerprint'],
        diagnostic_stops='meanL1>max(.1,4*historyMedian), preclipNorm>1, probeL1>2*prior and +.001; no skipping or rescue',
        scope='Native replay and first-warning capture only. Not a stability repair or continued training.',
        code_sha256=trainer.digest(Path(__file__)))
    protocol['model_code_sha256']=model_hashes
    write_json(out/'protocol.json',protocol)
    baseline_probe = probe(model,fixed_probe)
    assert rng()['python']==source_rng['python']
    matched, maximum_difference = 0, 0.
    status='MAXIMUM_REPLAY_REACHED_NO_WARNING'
    with (out/'observations.jsonl').open('w', buffering=1) as stream:
        for update in range(295001,295701):
            before = snapshot(model,optimizer,scheduler,update-1,rng())
            batches, samples_log = [], []
            def draw_batches():
                # Draw each microbatch immediately before its forward, as in the original trainer.
                for micro in range(4):
                    samples=[]
                    for _ in range(16):
                        pair=random.choice(train); trace=RandomTrace(random)
                        samples.append(trainer.sample(pair,192,trace))
                        samples_log.append(dict(micro=micro,hr=str(pair[0]),lr=str(pair[1]),draws=trace.draws))
                    cpu_batch=tuple(torch.stack([s[k] for s in samples]) for k in (0,1))
                    batches.append(cpu_batch)
                    yield tuple(t.cuda() for t in cpu_batch)
            feature_trace.rows=[]; feature_trace.enabled=update==295001 or update>=295300
            probe_before=probe(model,fixed_probe)
            feature_trace.rows=[]
            lr_used=[g['lr'] for g in optimizer.param_groups]
            try:
                value=step(model,optimizer,scheduler,draw_batches())
            except Exception as error:
                torch.save(before,out/'before.pt')
                torch.save(snapshot(model,optimizer,scheduler,update-1,rng()),out/'failed-after.pt')
                torch.save(dict(batches=batches,samples=samples_log,fixed_probe=fixed_probe,cfg=cfg),out/'batches.pt')
                write_json(out/'activation.json',feature_trace.rows)
                write_json(out/'report.json',dict(status='STEP_FAILED_WITH_STATE_CAPTURED',
                    attempted_update=update,error=repr(error),matched_updates=matched,
                    source_unchanged=trainer.digest(checkpoint)==SOURCE_SHA))
                raise
            feature_trace.enabled=False
            reason=[]
            if value.get('original_loss_guard'):
                status='ORIGINAL_LOSS_GUARD_REPRODUCED'; reason.append('original_micro_loss_guard')
                probe_after=None
            else:
                probe_after=probe(model,fixed_probe)
                difference=abs(value['mean_l1']-references[update]['train_l1'])
                maximum_difference=max(maximum_difference,difference)
                if difference>1e-7:
                    status='HISTORY_DIVERGENCE'; reason.append('historical_loss_mismatch')
                else:
                    matched+=1
                    if value['mean_l1']>max(.1,4*statistics.median(history)):
                        reason.append('training_loss_warning')
                    if value['gradient_norm_before_clip']>1.:
                        reason.append('gradient_warning')
                    if probe_after>2*probe_before and probe_after-probe_before>.001:
                        reason.append('same_probe_jump_after_adam')
                    if reason: status='WARNING_ON_MATCHED_TRAJECTORY'
                history.append(value['mean_l1']); history=history[-1000:]
            gradients=value.pop('gradients_after_clip',None)
            changes=value.pop('changes',[])
            row=dict(update=update,lr_used=lr_used,scheduler_last_epoch=scheduler.last_epoch,
                fixed_probe_before=probe_before,fixed_probe_after=probe_after,
                reference_l1=references[update]['train_l1'],reasons=reason,**value,
                top_updates=sorted(changes,key=lambda x:-x['update_norm'])[:12],
                top_relative_updates=sorted(changes,key=lambda x:-x['relative_update'])[:12])
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            if update%25==0 or reason or update==295001:
                print('REPLAY',update,'L1',value.get('mean_l1'),'postclip',value.get('gradient_norm_after_clip'),
                      'probe',probe_before,probe_after,'reason',reason,flush=True)
            if reason or update==295700:
                torch.save(before,out/'before.pt')
                torch.save(snapshot(model,optimizer,scheduler,update if probe_after is not None else update-1,rng()),out/'after.pt')
                torch.save(dict(batches=batches,samples=samples_log,gradients_after_clip=gradients,
                                fixed_probe=fixed_probe,cfg=cfg),out/'batches.pt')
                write_json(out/'activation.json',feature_trace.rows)
                write_json(out/'transition.json',dict(row=row,all_parameter_updates=changes))
                break
            del before,batches,gradients,changes
    unchanged=trainer.digest(checkpoint)==SOURCE_SHA
    assert unchanged
    result=dict(status=status,matched_updates=matched,maximum_loss_difference=maximum_difference,
        first_capture_update=update,baseline_probe_l1=baseline_probe,source_unchanged=unchanged,
        scope=protocol['scope'])
    write_json(out/'report.json',result)
    for name in ('before.pt','after.pt','batches.pt','activation.json','transition.json','observations.jsonl','report.json','protocol.json'):
        with (out/name).open('rb') as f: os.fsync(f.fileno())
    print('DIAGNOSTIC_FINISHED',json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
