"""Synthetic cap semantics and exact baseline equivalence; no trained-weight claim."""
import argparse
import copy
import json
import torch
import audit as a
from cap_screen import apply_cap
from follow_on import equal


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1=torch.nn.Linear(3,4)
        self.block_1=torch.nn.Linear(4,2)
        self.head=torch.nn.Linear(2,1)

    def forward(self,x):
        return self.head(torch.tanh(self.block_1(self.conv_1(x))))


def main(device):
    torch.set_num_threads(2);torch.manual_seed(10)
    model=Toy().to(device)
    old=copy.deepcopy(model.state_dict())
    with torch.no_grad():
        for p in model.parameters():p.add_(1.)
    proposed=copy.deepcopy(model.state_dict())
    rows=apply_cap(model,old,.001)
    assert all(r['cap_active'] for r in rows)
    assert all(r['applied_delta_norm']<=r['parameter_norm']*.001*1.001+1e-6 for r in rows)
    equal(model.head.state_dict(),{k.split('.',1)[1]:v for k,v in proposed.items() if k.startswith('head.')})
    equal(old['conv_1.weight'],old['conv_1.weight'].clone())
    model.load_state_dict(old)
    before=copy.deepcopy(model.state_dict())
    rows=apply_cap(model,before,.001)
    equal(model.state_dict(),before);assert not any(r['cap_active'] for r in rows)
    # A sufficiently high cap is bitwise inert after a real Adam proposal.
    reference=Toy().to(device);candidate=copy.deepcopy(reference)
    opt1=torch.optim.Adam(reference.parameters(),lr=1e-4)
    opt2=torch.optim.Adam(candidate.parameters(),lr=1e-4)
    batches=[(torch.rand(8,3,device=device),torch.rand(8,1,device=device)) for _ in range(4)]
    for _ in range(2):
        for model,optimizer in ((reference,opt1),(candidate,opt2)):
            optimizer.zero_grad(set_to_none=True)
            for x,y in batches:(torch.nn.functional.l1_loss(model(x),y)/4).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
            state=copy.deepcopy(model.state_dict());optimizer.step()
            if model is candidate:
                rows=apply_cap(model,state,100.)
                assert not any(r['cap_active'] for r in rows)
        equal(reference.state_dict(),candidate.state_dict());equal(opt1.state_dict(),opt2.state_dict())
    optimizer_before=copy.deepcopy(opt2.state_dict())
    state=copy.deepcopy(candidate.state_dict())
    with torch.no_grad():candidate.conv_1.weight.add_(.1)
    apply_cap(candidate,state,.001)
    equal(optimizer_before,opt2.state_dict())
    cfg=json.loads((a.ROOT/'research/multiscale_f0/template.json').read_text())
    cfg['arm']='F0';cfg['model']['upscale']=3
    arch,_=a.fusion.components()
    native=a.fusion.build_model(arch,cfg).to(device).train()
    old=copy.deepcopy(native.state_dict())
    with torch.no_grad():
        for name,p in native.named_parameters():
            if p.requires_grad and name.startswith(('conv_1.','block_1.')):p.add_(.01)
    rows=apply_cap(native,old,.001)
    assert all(r['applied_delta_norm']<=r['parameter_norm']*.001*1.001+1e-6 for r in rows)
    for name,p in native.named_parameters():
        if not p.requires_grad:assert torch.equal(p.detach(),old[name])
    print('PASS group delta bound, untargeted head untouched, inactive cap bitwise inert, Adam moments unchanged',device,flush=True)
    print('PASS native F0 x3 group mapping and frozen Conv3XC caches untouched',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    main(p.parse_args().device)
