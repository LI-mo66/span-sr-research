"""CPU FP64 algebra check for the production grouping/batch transformation."""
import torch
from scan import isolated_scan, reference_scan


def channel_reference(u,delta,A,B,C,D,delta_bias=None,delta_softplus=False):
    if delta_bias is not None: delta=delta+delta_bias[None,:,None]
    if delta_softplus: delta=torch.nn.functional.softplus(delta)
    state=u.new_zeros(u.shape[0],u.shape[1],A.shape[1]);out=[]
    for t in range(u.shape[-1]):
        dt=delta[:,:,t,None]
        state=torch.exp(dt*A[None])*state+dt*B[:,:,:,t]*u[:,:,t,None]
        out.append((state*C[:,:,:,t]).sum(-1)+u[:,:,t]*D)
    return torch.stack(out,-1)


for batch in (1,2,4):
    torch.manual_seed(10)
    args=[torch.randn(*shape,dtype=torch.float64,requires_grad=True) for shape in
          ((batch,4,7),(batch,4,7),(4,3),(batch,1,3,7),(batch,1,3,7),(4,),(4,))]
    u,dt,A,B,C,D,bias=args
    expected=reference_scan(u,dt,A,B,C,D,delta_bias=bias,delta_softplus=True)
    actual=isolated_scan(channel_reference,u,dt,A,B,C,D,delta_bias=bias,delta_softplus=True)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    target=torch.randn_like(actual)
    g1=torch.autograd.grad((actual*target).sum(),args)
    g2=torch.autograd.grad((expected*target).sum(),args)
    for a,b in zip(g1,g2):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
print('PASS: production isolated helper, FP64 forward and all input gradients, batches 1/2/4')
