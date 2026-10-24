"""Synthetic fine-tuning mechanics; not real accuracy evidence."""
import argparse
import io
import json
import tempfile
from common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=p.parse_args();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    cfg=json.loads((ROOT/'research/spatial_gate6/configs/LG6-SiLU-SG6.json').read_text())
    cfg.update(hr_patch=32,effective_batch=2,micro_batch=1)
    arch,metrics=sg_entry.existing.fs_runtime.components()
    ROOT.joinpath('tmp').mkdir(exist_ok=True)
    fixture=Path(tempfile.mkdtemp(prefix='ft-verify-',dir=ROOT/'tmp'))
    import cv2
    gen=np.random.default_rng(123)
    hp,lp=fixture/'hr.png',fixture/'lr.png'
    cv2.imwrite(str(hp),gen.integers(0,256,(48,64,3),dtype=np.uint8))
    cv2.imwrite(str(lp),gen.integers(0,256,(12,16,3),dtype=np.uint8))
    seed(10);base=build_model(arch,cfg).to(args.device)
    # A nonzero source mimics an already-trained network, not only zero gates.
    with torch.no_grad():
        for i in range(1,7):getattr(base,f'block_{i}').spatial_gate.project.weight.normal_(0,.0001)
    initial=state_copy(base,arch)
    reports={}
    for objective in ('L1','MSE'):
        model=build_model(arch,cfg).to(args.device);model.load_state_dict(initial,strict=True)
        opt=torch.optim.Adam(trainable(model).values(),lr=1e-5)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=4,eta_min=1e-6)
        shadow={k:v.detach().clone() for k,v in trainable(model).items()}
        seed(10)
        first=step(model,opt,scheduler,shadow,[(hp,lp)],cfg,objective,args.device)
        # EMA is initialized at source, first step is not overwritten by raw model.
        for name,param in trainable(model).items():
            expected=initial[name].to(args.device)*.999+param.detach()*.001
            torch.testing.assert_close(shadow[name],expected,rtol=1e-6,atol=1e-7)
        buffer=io.BytesIO();torch.save(dict(model=state_copy(model,arch),optimizer=opt.state_dict(),
            scheduler=scheduler.state_dict(),ema=shadow,rng=rng_state()),buffer);buffer.seek(0)
        saved=torch.load(buffer,map_location=args.device,weights_only=False)
        draws=[]
        snapshots=[]
        for _ in range(3):
            draws.append(step(model,opt,scheduler,shadow,[(hp,lp)],cfg,objective,args.device))
            snapshots.append(state_copy(model,arch))
        final=state_copy(model,arch)
        final_shadow={k:v.clone() for k,v in shadow.items()}
        assert abs(opt.param_groups[0]['lr']-1e-6)<1e-15
        clone=build_model(arch,cfg).to(args.device);clone.load_state_dict(saved['model'],strict=True)
        other=torch.optim.Adam(trainable(clone).values(),lr=1e-5);other.load_state_dict(saved['optimizer'])
        schedule2=torch.optim.lr_scheduler.CosineAnnealingLR(other,T_max=4,eta_min=1e-6);schedule2.load_state_dict(saved['scheduler'])
        other_shadow={k:v.clone() for k,v in saved['ema'].items()}
        restore_rng(saved['rng'])
        for expected in draws:
            actual=step(clone,other,schedule2,other_shadow,[(hp,lp)],cfg,objective,args.device)
            assert actual==expected
        resumed=state_copy(clone,arch)
        assert all(torch.equal(v,resumed[k]) for k,v in final.items())
        assert all(torch.equal(v,other_shadow[k]) for k,v in final_shadow.items())
        assert scheduler.state_dict()==schedule2.state_dict()
        values=average(snapshots,trainable(clone))
        for k in values:
            expected=torch.stack([s[k].double() for s in snapshots]).mean(0).to(values[k].dtype)
            torch.testing.assert_close(values[k],expected,rtol=0,atol=0)
        # Nontrainable eval weights are deliberately corrupt: install must rebuild them.
        with torch.no_grad():
            for module in clone.modules():
                if isinstance(module,arch.Conv3XC):module.eval_conv.weight.fill_(float('nan'))
        install(clone,values,arch)
        result=deployment_check(clone,arch,args.device)
        score=evaluate(clone,[(hp,lp)],metrics,cfg,fixture/(objective+'.json'),args.device)
        assert all(np.isfinite(list(score.values())))
        reports[objective]=dict(first_step=first,deployment=result)
    print(json.dumps(dict(status='PASS',device=args.device,arms=reports,
        checks=['both objectives finite','exact EMA first update from R0','optimizer scheduler model EMA RNG resume equality',
                'cosine endpoint','average arithmetic','derived eval weights rebuilt','unchanged deployed parameter count',
                'full-image original metric pipeline'],fixture=str(fixture),evidence='Synthetic only; no real fine-tuning benefit measured'),indent=2))


if __name__=='__main__':main()
