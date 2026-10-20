"""Compare execution chunks against unchanged official attention classes."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from adapter import AttentionAdapter
from vendor.reference import ATD_CA, AC_MSA


class Reference(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(52)
        self.wqkv=nn.Linear(52,156)
        self.tdca=ATD_CA(52,num_tokens=128,reducted_dim=12)
        self.acmsa=AC_MSA(52,num_heads=4,category_size=128)
        self.dictionary=nn.Parameter(torch.randn(128,52))

    def forward(self,image):
        b,c,h,w=image.shape
        x=self.norm(image.flatten(2).transpose(1,2).contiguous())
        direct,weights=self.tdca(x,self.dictionary.unsqueeze(0).expand(b,-1,-1),(h,w))
        grouped=self.acmsa(self.wqkv(x),weights.argmax(-1),(h,w))
        return image+(direct+grouped).transpose(1,2).reshape_as(image)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu',choices=('cpu','cuda'))
    parser.add_argument('--output')
    args=parser.parse_args()
    torch.set_num_threads(2); torch.manual_seed(10)
    records=[]
    for shape,qc,gc in [
            *((s,37,2) for s in ((1,52,1,1),(1,52,1,7),(2,52,3,43),(1,52,13,17),(1,52,64,65))),
            ((2,52,13,17),10**9,10**9),((1,52,64,65),10**9,10**9)]:
        # Small chunks exercise both query and group partition boundaries.
        model=AttentionAdapter(query_chunk=qc,group_chunk=gc).to(args.device)
        reference=Reference().to(args.device)
        reference.load_state_dict(model.state_dict(),strict=True)
        x=torch.randn(*shape,device=args.device,requires_grad=True)
        xr=x.detach().clone().requires_grad_()
        y,yr=model(x),reference(xr)
        torch.testing.assert_close(y,yr,atol=2e-6,rtol=1e-5)
        target=torch.randn_like(y)
        ((y-target)**2).mean().backward(); ((yr-target)**2).mean().backward()
        torch.testing.assert_close(x.grad,xr.grad,atol=1e-7,rtol=1e-4)
        max_param_error=0.
        for (name,p),(other,q) in zip(model.named_parameters(),reference.named_parameters()):
            assert name==other and p.grad is not None and q.grad is not None
            torch.testing.assert_close(p.grad,q.grad,atol=2e-7,rtol=2e-3)
            max_param_error=max(max_param_error,float((p.grad-q.grad).abs().max()))
        records.append(dict(shape=shape,query_chunk=qc,group_chunk=gc,output_error=float((y-yr).detach().abs().max()),parameter_gradient_error=max_param_error))
    # Verify image isolation with original full sorting and boundary semantics.
    model=AttentionAdapter(query_chunk=37,group_chunk=2).double().to(args.device).eval()
    x=torch.randn(2,52,3,43,dtype=torch.float64,device=args.device)
    with torch.no_grad():
        together=model(x); separate=torch.cat([model(v[None]) for v in x])
    torch.testing.assert_close(together,separate,atol=1e-12,rtol=1e-12)
    frozen={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad(): model(x)
    assert all(torch.equal(v,frozen[k]) for k,v in model.state_dict().items())
    report=dict(status='PASS',device=args.device,reference_commit='c064084c3a33724e850ec03532fc6ddc7d7a0828',
                cases=records,batch_isolation_fp64=float((together-separate).abs().max()),
                evidence='attention extraction/chunk semantics only; no full ATD or PSNR claim')
    if args.output: Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
