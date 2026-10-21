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


def scan(backend, *args, **kwargs):
    if backend == 'reference':
        return reference_scan(*args, **kwargs)
    if backend != 'cuda':
        raise ValueError('Unknown selective scan backend')
    if not args[0].is_cuda:
        raise ValueError('Production scan requires CUDA input')
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except ImportError as exc:
        raise RuntimeError('Compatible mamba_ssm CUDA kernel required; no reference fallback') from exc
    return selective_scan_fn(*args, **kwargs)
