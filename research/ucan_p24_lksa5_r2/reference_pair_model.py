"""Preserve SG6 keys, tuple semantics, image bypass, and fusion taps."""
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
for directory in ('spatial_gate6', 'gated_bypass', 'linear_gate'):
    sys.path.insert(1, str(HERE.parent / directory))
from sg_model import build_model as build_sg6, validate_variant as validate_sg6
from attention import ContextAdapter

VERSION = 'SG6-UCAN-QKPair-v2'
SETTINGS = dict(positions=[2, 4], channels=52, heads=1, norm='branch-pre-ln',
                projection_init='zero', fourier_floor=1e-4,
                fourier_normalization='abs_sum', writeback_scale='detached-image-rms',
                position_encoding='source-flattened-1d', detach_qk=False)


def validate_variant(cfg):
    validate_sg6(cfg)
    settings = cfg.get('ucan_pair', {})
    mode = settings.get('mode')
    if mode not in ('off', 'independent', 'reuse') or settings != dict(SETTINGS, mode=mode):
        raise ValueError('Unknown UCAN pair configuration')
    if any(cfg.get(k) for k in ('sgla', 'atd_attention', 'easa', 'cata_tab', 'cata_ffn', 'variants')):
        raise ValueError('Mixed candidates are forbidden')
    return mode


def build_model(arch, cfg):
    mode = validate_variant(cfg)
    model = build_sg6(arch, cfg)
    if mode == 'off':
        return model
    # Generate both independent adapters first so all common tensors and RNG
    # streams are identical in independent/reuse. Then remove unused Q/K.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg['seed'] + 80001)
        model.context_2 = ContextAdapter()
        model.context_4 = ContextAdapter()
    if mode == 'reuse':
        del model.context_4.core.q
        del model.context_4.core.k
        model.context_4.core.reuse = True

    class ContextPairSPAN(type(model)):
        def forward(self, image):
            # Exact GB/SG6 forward, with only the two explicit insertions.
            # Q/K is a local variable, never a module attribute or global cache.
            residual = self.image_residual(image)
            gate = 2.0 * torch.sigmoid(self.image_gate(image))
            self.mean = self.mean.type_as(image)
            x = (image - self.mean) * self.img_range
            feature = self.conv_1(x)
            b1, _, _ = self.block_1(feature)
            b2, _, _ = self.block_2(b1)
            b2, state = self.context_2(b2)
            b3, _, _ = self.block_3(b2)
            b4, _, _ = self.block_4(b3)
            b4, _ = self.context_4(b4, state if mode == 'reuse' else None)
            b5, _, _ = self.block_5(b4)
            b6, first, _ = self.block_6(b5)
            b6 = self.conv_2(b6)
            fused = self.conv_cat(torch.cat((feature, b6, b1, first), dim=1))
            pixels = self.upsampler[0](fused)
            return self.upsampler[1](pixels + gate * residual)

    model.__class__ = ContextPairSPAN
    extra = sum(p.numel() for name, p in model.named_parameters() if name.startswith('context_'))
    if extra > 535772 // 10:
        raise ValueError('Candidate exceeds 10% of fixed SG6 deployed parameters')
    model.ucan_extra_parameters = extra
    model.extra_parameters += extra
    return model
