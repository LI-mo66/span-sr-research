"""Same signed C1 feature bound used by the x4 bound8 recovery."""
import torch

def bounded(x,knee=8.):
    a=x.abs()
    return torch.where(a<=knee,x,x.sign()*(knee+knee*torch.tanh((a-knee).clamp_min(0)/knee)))
