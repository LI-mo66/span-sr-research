"""Eight D4 transforms, exact inverse, FP32 average before clipping/quantization."""
import torch

TRANSFORMS=tuple((transpose,vertical,horizontal) for transpose in (False,True)
                 for vertical in (False,True) for horizontal in (False,True))

def transform(x,code,inverse=False):
    transpose,vertical,horizontal=code
    if inverse:
        if horizontal:x=x.flip(-1)
        if vertical:x=x.flip(-2)
        if transpose:x=x.transpose(-2,-1)
    else:
        if transpose:x=x.transpose(-2,-1)
        if vertical:x=x.flip(-2)
        if horizontal:x=x.flip(-1)
    return x.contiguous()

def predict(model,x):
    plain=None;accumulator=None
    for index,code in enumerate(TRANSFORMS):
        value=transform(model(transform(x,code)),code,inverse=True).float()
        if value.shape != (x.shape[0],3,x.shape[-2]*4,x.shape[-1]*4):
            raise ValueError('x8 inverse/output shape mismatch')
        if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite x8 prediction')
        if index==0:
            plain=value
            accumulator=value.clone()
        else:accumulator.add_(value)
    return plain,accumulator.div_(8)
