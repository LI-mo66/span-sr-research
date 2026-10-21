"""Selective scan: strict CUDA production backend and small diagnostic reference.

Equation adapted from MambaIRv2 / Mamba; see vendor/NOTICE.md.
The reference is intentionally bounded and cannot silently run a 50K job.
"""
import torch
from torch.nn import functional as F


def reference_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                   delta_softplus=False, return_last_state=False):
    if u.shape[-1] > 4096:
        raise ValueError('Reference scan is diagnostic-only, maximum 4096 tokens')
    if B.shape[1] != 1 or C.shape[1] != 1 or z is not None:
        raise ValueError('This diagnostic reference supports one group and no z')
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    result = []
    for i in range(u.shape[-1]):
        dt = delta[:, :, i, None]
        state = torch.exp(dt * A[None]) * state + dt * B[:, 0, :, i][:, None] * u[:, :, i, None]
        value = (state * C[:, 0, :, i][:, None]).sum(-1)
        if D is not None:
            value = value + u[:, :, i] * D
        result.append(value)
    output = torch.stack(result, -1)
    return (output, state) if return_last_state else output


def isolated_scan(native, u, delta, A, B, C, D=None, z=None,
                  delta_bias=None, delta_softplus=False, return_last_state=False):
    """Same recurrence, batch-one calls and explicit channel groups for B/C.

    Avoid shared batch/channel atomic accumulation in the tested Mamba 2.2.2
    backward kernel. Extra B/C storage is intentional and must be budgeted.
    This ASSM path does not use z or returned final states.
    """
    if z is not None or return_last_state:
        raise ValueError('Isolated ASSM scan supports no z or returned state')
    if B.ndim != 4 or C.ndim != 4 or B.shape[1] != 1 or C.shape[1] != 1:
        raise ValueError('Isolated ASSM scan expects single-group variable B/C')
    if u.shape[0] < 1 or B.shape[0] != u.shape[0] or C.shape[0] != u.shape[0]:
        raise ValueError('Invalid isolated scan batch')
    outputs = []
    for i in range(u.shape[0]):
        b = B[i:i+1].expand(-1, u.shape[1], -1, -1).contiguous()
        c = C[i:i+1].expand(-1, u.shape[1], -1, -1).contiguous()
        outputs.append(native(u[i:i+1], delta[i:i+1], A, b, c, D,
                              delta_bias=delta_bias, delta_softplus=delta_softplus))
    return torch.cat(outputs, 0)


def scan(backend, *args, **kwargs):
    if backend == 'reference':
        return reference_scan(*args, **kwargs)
    if backend not in ('cuda', 'cuda_isolated'):
        raise ValueError('Unknown selective scan backend')
    if not args[0].is_cuda:
        raise ValueError('Production scan requires CUDA input')
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except ImportError as exc:
        raise RuntimeError('Compatible mamba_ssm CUDA kernel required; no reference fallback') from exc
    if backend == 'cuda_isolated':
        return isolated_scan(selective_scan_fn, *args, **kwargs)
    return selective_scan_fn(*args, **kwargs)
