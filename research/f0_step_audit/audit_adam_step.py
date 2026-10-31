"""Independent saved-batch Adam-step replay; never edits source checkpoints."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, copy, hashlib, json, random, sys
from pathlib import Path
import numpy as np
import torch


def compare(a, b, path='root', differences=None):
    if differences is None: differences=[]
    if torch.is_tensor(a) and torch.is_tensor(b):
        a,b=a.detach().cpu(),b.detach().cpu()
        if a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a,b):
            error=float((a.double()-b.double()).abs().max()) if a.shape==b.shape and a.numel() else None
            differences.append(dict(path=path,max_error=error))
    elif isinstance(a,dict) and isinstance(b,dict):
        if a.keys()!=b.keys():differences.append(dict(path=path,error='keys'))
        for k in a.keys() & b.keys():compare(a[k],b[k],f'{path}.{k}',differences)
    elif isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):
        if len(a)!=len(b):differences.append(dict(path=path,error='length'))
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}.{i}',differences)
    elif a!=b:differences.append(dict(path=path,error='value'))
    return differences


def step(model,opt,sched,batches,device):
    model.train();opt.zero_grad(set_to_none=True)
    values=[]
    for samples in batches:
        low=torch.stack([s[0] for s in samples]).to(device)
        high=torch.stack([s[1] for s in samples]).to(device)
        assert low.ndim==4 and high.shape[-2:]==tuple(v*4 for v in low.shape[-2:])
        assert bool(torch.isfinite(low).all() and torch.isfinite(high).all())
        assert float(low.min())>=0 and float(low.max())<=1
        assert float(high.min())>=0 and float(high.max())<=1
        loss=torch.nn.functional.l1_loss(model(low),high)/len(batches)
        values.append(float(loss.detach())*len(batches));loss.backward()
    gradients=sorted([(float(p.grad.abs().max()),n) for n,p in model.named_parameters()
                      if p.grad is not None],reverse=True)[:20]
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
               for p in model.parameters() if p.requires_grad)
    opt.step();sched.step()
    return dict(micro_losses=values,mean_l1=sum(values)/len(values),gradient_top=gradients)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root,out=args.project,args.output
    out.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    sys.path.insert(0,str(root/'research/fusion_wu'));import fusion
    folder=root/'logs/f0-original-lr-first-transition'
    paths=[folder/n for n in ('transition-before.pt','transition-after.pt','trigger-training-batches.pt')]
    hashes={str(f):digest(f) for f in paths}
    before,after,batches=[torch.load(f,map_location='cpu',weights_only=False) for f in paths]
    assert before['update']==99728 and after['update']==99729
    cfg=json.loads((root/'logs/f0-long/frozen/config.json').read_text())
    assert len(batches)==cfg['effective_batch']//cfg['micro_batch']==4
    assert all(len(b)==cfg['micro_batch']==16 for b in batches)
    arch,_=fusion.components();model=fusion.build_model(arch,cfg).cuda()
    opt=torch.optim.Adam((v for v in model.parameters() if v.requires_grad),lr=cfg['lr'],
        betas=tuple(cfg['adam_betas']),eps=cfg['adam_eps'])
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,cfg['milestones'],gamma=cfg['gamma'])
    model.load_state_dict(before['model'],strict=True)
    opt.load_state_dict(before['optimizer']);sched.load_state_dict(before['scheduler'])
    random.setstate(before['python_rng']);np.random.set_state(before['numpy_rng'])
    torch.set_rng_state(before['torch_rng']);torch.cuda.set_rng_state_all(before['cuda_rng'])
    measured=step(model,opt,sched,batches,'cuda')
    differences={label:compare(value,after[label],label) for label,value in (
        ('model',model.state_dict()),('optimizer',opt.state_dict()),('scheduler',sched.state_dict()))}
    total={k:len(v) for k,v in differences.items()}
    reference=None
    for line in (root/'runs/FEAT-SG6-F0-X8-S10-1M-01/loss_steps.jsonl').open():
        row=json.loads(line)
        if row['update']==99729:reference=row;break
    report=dict(status='EXACT_MATCH' if not any(total.values()) else 'MISMATCH',
        measured=measured,original_step=reference,difference_counts=total,
        first_differences={k:v[:12] for k,v in differences.items()},
        source_hashes=hashes,source_unchanged=all(digest(Path(k))==v for k,v in hashes.items()),
        scope='Exact saved-batch update replay, not proof of global numerical stability')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    assert report['source_unchanged']


if __name__=='__main__':main()

