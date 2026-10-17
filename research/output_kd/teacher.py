"""Frozen official classical x4 teacher; no change to the student state dict."""
import hashlib
import importlib.util
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
REVISION = '6545850fbf8df298df73d81f3e8cba638787c8bd'
SOURCE_SHA256 = '9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913'
WEIGHT_NAME = '001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
WEIGHT_URL = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/' + WEIGHT_NAME


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def architecture():
    path = HERE / 'vendor/network_swinir.py'
    if sha256(path) != SOURCE_SHA256:
        raise ValueError('Pinned teacher source hash mismatch')
    spec = importlib.util.spec_from_file_location('output_kd_official_swinir', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SwinIR


def build_teacher(cfg, device):
    settings = cfg['distillation']
    path = Path(settings['checkpoint'])
    actual = sha256(path)
    if actual != settings['checkpoint_sha256']:
        raise ValueError('Teacher checkpoint SHA256 mismatch')
    # Constructing the teacher must not advance the student's/data RNG streams.
    with torch.random.fork_rng(devices=[]):
        net = architecture()(upscale=4, in_chans=3, img_size=64, window_size=8,
                             img_range=1., depths=[6]*6, embed_dim=180,
                             num_heads=[6]*6, mlp_ratio=2, upsampler='pixelshuffle',
                             resi_connection='1conv')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or 'params' not in state:
            raise ValueError('Expected official SwinIR checkpoint with params key')
        net.load_state_dict(state['params'], strict=True)
    net.requires_grad_(False).eval().to(device)
    return net


@torch.no_grad()
def teacher_prediction(teacher, lr, chunk_size=1):
    if chunk_size < 1 or lr.ndim != 4 or lr.shape[1] != 3:
        raise ValueError('Invalid teacher batch')
    teacher.eval()
    result = torch.cat([teacher(x) for x in lr.split(chunk_size)], dim=0)
    expected = (lr.shape[0], 3, lr.shape[2]*4, lr.shape[3]*4)
    if result.shape != expected or not torch.isfinite(result).all():
        raise ValueError('Invalid teacher output shape or nonfinite output')
    return result


def output_loss(prediction, hr, target, weight):
    if prediction.shape != hr.shape or target.shape != prediction.shape:
        raise ValueError('Output distillation requires exact RGB/spatial alignment')
    if not weight > 0:
        raise ValueError('Distillation weight must be positive')
    reconstruction = F.l1_loss(prediction, hr)
    distillation = F.l1_loss(prediction, target.detach())
    return reconstruction + weight*distillation, reconstruction, distillation
