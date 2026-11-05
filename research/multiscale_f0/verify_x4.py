"""Check scale-generalized bound8 mapping against frozen x4 implementation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(1,str(ROOT/'research'))
import fusion
from bound import bounded

def main():
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    arch,_=fusion.components();cfg=json.loads((HERE/'template.json').read_text());cfg['arm']='F0'
    cfg['model']['upscale']=4
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        torch.manual_seed(10);candidate=fusion.build_model(arch,cfg).to(device)
        reference=copy.deepcopy(candidate)
        # Replace only the bound placements with archived unbounded formulas,
        # then install the exact server feature hooks again.
        for i in range(1,7):
            wrapper=getattr(reference,f'block_{i}');block=getattr(wrapper,'original',wrapper)
            def unbounded(self,x):
                first=self.c1_r(x);a=self.act1(first)
                second=self.c2_r(a);a=self.act1(second)
                third=self.c3_r(a)
                gate=self.local_gate(third)+self.spatial_gate(third)
                return (third+x)*gate,first,gate
            import types
            block.forward=types.MethodType(unbounded,block)
            def main_hook(module,inputs,outputs):
                return bounded(outputs[0],8.),outputs[1],outputs[2]
            block.register_forward_hook(main_hook)
        def lksa(self,x):
            scale=x.detach().square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-6)
            return x+scale*self.project(self.core(x/scale))
        reference.block_5.adapter.forward=types.MethodType(lksa,reference.block_5.adapter)
        def lksa_hook(module,inputs,output):return bounded(inputs[0]+bounded(output-inputs[0],8.),8.)
        reference.block_5.adapter.register_forward_hook(lksa_hook)
        # Nonzero learned-style LKSA output projection to exercise writeback.
        with torch.no_grad():
            candidate.block_5.adapter.project.weight.normal_(0,.02)
        reference.load_state_dict(candidate.state_dict(),strict=True)
        for h,w in ((3,5),(17,19)):
            x=torch.rand(1,3,h,w,device=device)
            with torch.no_grad():
                a=candidate(x);b=reference(x)
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        print('PASS x4 server-bound mapping',device,flush=True)
if __name__=='__main__':main()
