import argparse,copy,json,os,torch
from runtime import ROOT,components
from modules.candidates import build_model
from deploy import fused_copy

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=p.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2);arch,_=components()
    cfg=json.loads((ROOT/'research/configs/baseline_x4.json').read_text())
    torch.manual_seed(cfg['seed']);baseline=build_model(arch,cfg).to(args.device)
    expected_cpu_rng=torch.get_rng_state();expected_cuda_rng=torch.cuda.get_rng_state_all()
    cases=[[dict(name='fusion',position='fusion')],[dict(name='context',position='mid')],[dict(name='context',position='tail')],[dict(name='frequency',position='mid')],[dict(name='frequency',position='tail')],[dict(name='fusion',position='fusion'),dict(name='context',position='tail'),dict(name='frequency',position='mid')]]
    for variants in cases:
        option=copy.deepcopy(cfg);option['variants']=variants;torch.manual_seed(cfg['seed']);model=build_model(arch,option).to(args.device)
        assert torch.equal(expected_cpu_rng,torch.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(expected_cuda_rng,torch.cuda.get_rng_state_all()))
        assert all(torch.equal(value,model.state_dict()[key]) for key,value in baseline.state_dict().items())
        optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=5e-4)
        x=torch.rand(2,3,13,17,device=args.device);hr=torch.rand(2,3,52,68,device=args.device)
        if variants==cases[0]:
            with torch.no_grad():torch.testing.assert_close(model(x),baseline(x),atol=0,rtol=0)
        before={k:p.detach().clone() for k,p in model.adapters.named_parameters()}
        for _ in range(3):
            optimizer.zero_grad();loss=(model(x)-hr).abs().mean();assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.adapters.parameters())
            optimizer.step()
        assert all(not torch.equal(before[k],p) for k,p in model.adapters.named_parameters())
        model.eval();deployed=fused_copy(model,arch.Conv3XC);clone=copy.deepcopy(model);clone.load_state_dict(model.state_dict(),strict=True)
        with torch.no_grad():
            y=model(x);torch.testing.assert_close(y,deployed(x),atol=1e-6,rtol=1e-5);torch.testing.assert_close(y,clone(x),atol=0,rtol=0)
        print(dict(variants=variants,extra=model.extra_parameters,deployed=sum(p.numel() for p in deployed.parameters()),status='PASS'),flush=True)
    print('Engineering evidence only: no candidate PSNR gain measured.')
if __name__=='__main__':main()
