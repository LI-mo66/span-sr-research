"""Shared I/O, pinned official model factories, and invariants for the SCI final visuals suite."""
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
import torch


BASE = Path(os.environ.get('PAPER_BASE', '/home/featurize/work/span-x4-research'))
PROJECT = Path(os.environ.get('PAPER_PROJECT', str(BASE / 'code/span-sg6-f0-long-x8-1m')))
ROOT = Path(os.environ.get('PAPER_ROOT', str(BASE / 'records/paper_visuals_20261126')))
OLD_OUT = Path(os.environ.get('PAPER_OLD_OUT', str(ROOT / 'step10-paper-suite-v2-01')))
OUT = Path(os.environ.get('PAPER_OUT', str(ROOT / 'step11-paper-sci-final-01')))
DATA = Path(os.environ.get('PAPER_DATA', '/home/featurize/datasets/benchmark'))

DD_COMMIT = 'b34bb97ba68f91a25afee23d737d97ca0a270bde'
IMDN_COMMIT = '8f158e6a5ac9db6e5857d9159fd4a6c4214da574'
LBNET_COMMIT = '0fe8632f74444332179df030ce40b560bc4902b9'
SMFA_COMMIT = 'a05e0bb554bbbb80cd27d8fd2b78d4228c657170'
LAM_COMMIT = '6e75cc026ee8570f3966a6f017e9152556400876'

FINAL_SHA = '1f691364ec3958db9eb67800c583e032c1813d8ddd8d616b466f4717600d7448'
BOUND_SHA = '19bef1a8a112c5cb6136478c0130fab9466abdd80987c4f336a3ffbd133f16fb'

ALL_DATASETS = [('Set5', 5), ('Set14', 14), ('B100', 100), ('Urban100', 100), ('Manga109', 109)]

RUNS = {
    'B0': ('d280dc5ebcffc46c50983dfe27e81c6096d90dcf',
           'FEAT-MOEFS-G4-LR2E4-20K-01-1791983354063378494/runs/FEAT-MOEFS-G4-LR2E4-20K-01-B0-s10'),
    'SG6': ('e731483d4ce4e325f3d64dbbc80add342b46f7f1',
            'FEAT-LG6-SG6-50K-01-1792264356078917269/runs/FEAT-LG6-SG6-50K-01-LG6-SiLU-SG6-s10'),
    'W': ('78cc1dda90a5bafcefa6c62225197f6eb95c92e2',
          'FEAT-SG6-MID-W16-PC26-AC-LR2E4-50K-01-INTERRUPTED-20261026T033650Z-20587/runs/FEAT-SG6-MID-W16-PC26-AC-LR2E4-50K-01-A-s10'),
    'U': ('d42b73fef5b2ec3e088f340ab033a5c4ac13a190',
          'FEAT-SG6-UCAN-P24-LKSA5-V1-50K-01-20261025T125002Z/runs/FEAT-SG6-UCAN-P24-LKSA5-V1-50K-01'),
    'F0': ('2416cd660f25c3a8351330a8ad56d7fc8aa233e0',
           'FEAT-SG6-W-UCAN-P24-LKSA5-FUSION-V1-50K-01-1793142163941762977/runs/FEAT-SG6-W-UCAN-P24-LKSA5-FUSION-V1-50K-01'),
}

ABLATION_ENTRIES = {
    'W': ('research/sg6_mid_window/model.py', 'b5aa9d8d24aa2dd4143ff70040bd70587c64d063'),
    'U': ('research/ucan_p24_lksa5/candidate.py', '7aa5cf8c6e3f4f86d4bfc458c3aba9130835a6de'),
    'F0': ('research/fusion_wu/fusion.py', 'ab57d5a5e3a87bb7419a52497e29aa2733a6de0b'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, obj, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists() and not overwrite:
        if path.read_text(encoding='utf-8') == data:
            return
        raise FileExistsError(f'Refusing to overwrite existing evidence: {path}')
    temporary = path.with_name(path.name + '.part')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def fetch(url, path, blob=None, expected_sha=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Reuse already downloaded external files from OLD_OUT if present.
        try:
            rel = path.relative_to(OUT)
            alt = OLD_OUT / rel
            if alt.is_file():
                path.write_bytes(alt.read_bytes())
        except ValueError:
            pass
    if not path.exists():
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'SPAN-paper-sci-final'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    raw = response.read(100_000_001)
                if len(raw) > 100_000_000:
                    raise ValueError('Unexpected oversized download')
                if raw.startswith(b'<!DOCTYPE html') or raw.startswith(b'<html'):
                    raise ValueError('HTML instead of asset')
                part = path.with_name(path.name + '.part')
                part.write_bytes(raw)
                part.replace(path)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    raw = path.read_bytes()
    if blob and hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() != blob:
        raise ValueError(f'Official git blob mismatch: {path}')
    if expected_sha and sha(path) != expected_sha:
        raise ValueError(f'SHA256 mismatch: {path}')
    return dict(path=str(path), bytes=len(raw), sha256=sha(path), url=url, git_blob=blob)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime(project=PROJECT):
    return load_module('paper_runtime', Path(project) / 'research/runtime.py').components()


_metric_module = None


def metrics(sr, gt, project=PROJECT):
    global _metric_module
    if _metric_module is None:
        _, _metric_module = runtime(project)
    a, b = sr[:, :, ::-1].copy(), gt[:, :, ::-1].copy()
    return dict(
        psnr=float(_metric_module.calculate_psnr(a, b, crop_border=4, test_y_channel=True)),
        ssim=float(_metric_module.calculate_ssim(a, b, crop_border=4, test_y_channel=True)),
    )


def dataset(name):
    folders = [p for p in DATA.iterdir() if p.is_dir() and p.name.casefold() == name.casefold()]
    if len(folders) != 1:
        raise ValueError(f'Expected one dataset directory for {name} under {DATA}')
    return folders[0]


def pair(name, image):
    root = dataset(name)
    hp = root / 'HR' / image
    lp = root / 'LR_bicubic/X4' / (Path(image).stem + 'x4.png')
    hr = np.array(Image.open(hp).convert('RGB'))
    lr = np.array(Image.open(lp).convert('RGB'))
    h, w = lr.shape[0] * 4, lr.shape[1] * 4
    if not (0 <= hr.shape[0] - h < 4 and 0 <= hr.shape[1] - w < 4):
        raise ValueError(f'HR/LR alignment mismatch: {hp} vs {lp}')
    return hr[:h, :w].copy(), lr, dict(hr=str(hp), lr=str(lp), hr_sha256=sha(hp), lr_sha256=sha(lp))


def tensor(rgb, device):
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().unsqueeze(0).to(device) / 255.0


def quantize(output):
    if output.ndim != 4 or output.shape[:2] != (1, 3) or not torch.isfinite(output).all():
        raise ValueError('Invalid or non-finite RGB prediction')
    return np.rint(output[0].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)


def save_png(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not np.array_equal(np.array(Image.open(path).convert('RGB')), rgb):
            raise ValueError(f'Existing PNG differs: {path}')
    else:
        temporary = path.with_name(path.stem + '.part.png')
        Image.fromarray(rgb).save(temporary, format='PNG')
        if not np.array_equal(np.array(Image.open(temporary).convert('RGB')), rgb):
            raise ValueError('Temporary PNG readback mismatch')
        temporary.replace(path)
    if not np.array_equal(np.array(Image.open(path).convert('RGB')), rgb):
        raise ValueError('PNG readback mismatch')


def image_path(ds, method, image, extra_roots=None):
    roots = [OUT / 'sr_outputs', OLD_OUT / 'sr_outputs', ROOT / 'sr_outputs']
    if extra_roots:
        roots = [Path(r) for r in extra_roots] + roots
    present = []
    for base in roots:
        p = base / ds / method / image
        if p.is_file():
            present.append(p)
    if not present:
        raise FileNotFoundError(f'Missing PNG for ({ds}, {method}, {image}) across {roots}')
    if len(present) > 1:
        h0 = sha(present[0])
        for p in present[1:]:
            if sha(p) != h0:
                raise ValueError(f'Conflicting PNGs for ({ds}, {method}, {image}): {present[0]} vs {p}')
    return present[0]


def find_existing_png(ds, method, image):
    try:
        return image_path(ds, method, image)
    except FileNotFoundError:
        return None


def get_image(ds, method, image, extra_roots=None):
    return np.array(Image.open(image_path(ds, method, image, extra_roots=extra_roots)).convert('RGB'))


def fused_copy(model, conv_type):
    result = copy.deepcopy(model).eval()

    def visit(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, conv_type):
                child.update_params()
                if child.has_relu:
                    raise ValueError('Export needs explicit activation handling for this Conv3XC')
                setattr(parent, name, copy.deepcopy(child.eval_conv))
            else:
                visit(child)

    visit(result)
    return result


def final_model(device, deploy_fold=False):
    cfg = read_json(PROJECT / 'logs/f0-finetune-five-01/FT2/config.json')
    checkpoint = BASE / 'records/F0-X4-FT2-20K-FINAL-ARCHIVE-01/assets/F0-x4-FT2-20K-full.pt'
    boundpath = BASE / 'records/F0-FEATURE-BOUND8-1M-FINAL-01/candidate/feature_bound.py'
    if sha(checkpoint) != FINAL_SHA or sha(boundpath) != BOUND_SHA:
        raise ValueError('Final weight/bound hash changed')
    fusion = load_module('paper_fusion_final', PROJECT / 'research/fusion_wu/fusion.py')
    arch, _ = runtime(PROJECT)
    model = fusion.build_model(arch, cfg)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['update'] != 20000:
        raise ValueError('Wrong finetune update')
    model.load_state_dict(state['model'], strict=True)
    if deploy_fold:
        model = fused_copy(model, arch.Conv3XC)
    load_module('paper_bound8', boundpath).install(model, 8.0)
    return model.to(device).eval(), dict(
        path=str(checkpoint),
        sha256=sha(checkpoint),
        weight_key='model',
        bound8=str(boundpath),
        bound_sha256=sha(boundpath),
        deploy_folded=deploy_fold,
        parameters=sum(p.numel() for p in model.parameters()),
    )


def structural_ours_x2_deploy(device):
    """Build structural x2 deployed-folded F0 model for theoretical Params/FLOPs & runtime check."""
    cfg = copy.deepcopy(read_json(PROJECT / 'logs/f0-finetune-five-01/FT2/config.json'))
    cfg['model']['upscale'] = 2
    fusion = load_module('paper_fusion_final', PROJECT / 'research/fusion_wu/fusion.py')
    arch, _ = runtime(PROJECT)
    model = fusion.build_model(arch, cfg)
    model = fused_copy(model, arch.Conv3XC)
    boundpath = BASE / 'records/F0-FEATURE-BOUND8-1M-FINAL-01/candidate/feature_bound.py'
    load_module('paper_bound8', boundpath).install(model, 8.0)
    return model.to(device).eval(), dict(
        parameters=sum(p.numel() for p in model.parameters()),
        scope='structural_deploy_only_no_x2_checkpoint_on_server',
    )


def check_ablation_entry(name, src):
    relative, expected = ABLATION_ENTRIES[name]
    path = Path(src) / relative
    raw = path.read_bytes()
    observed = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if observed != expected:
        raise ValueError('Archived factory blob mismatch: ' + str(path))
    return path, observed


def ensure_ablation_source(name):
    commit, _ = RUNS[name]
    for base_out in (OLD_OUT, OUT):
        src = base_out / 'source50' / name
        marker = src / 'SOURCE_COMMIT.txt'
        if marker.is_file() and marker.read_text(encoding='utf-8').strip() == commit:
            return src
    src = OUT / 'source50' / name
    src.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(['git', '-C', str(PROJECT), 'cat-file', '-e', commit + '^{commit}'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        subprocess.run(['git', '-C', str(PROJECT), 'fetch', 'origin', commit], check=True)
    import tarfile
    archive = OUT / f'source-{name}.tar'
    subprocess.run(['git', '-C', str(PROJECT), 'archive', '--format=tar', '--output', str(archive), commit], check=True)
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile():
                continue
            target = src / member.name
            if not target.resolve().is_relative_to(src.resolve()):
                raise ValueError('Unsafe source archive member')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(tar.extractfile(member).read())
    (src / 'SOURCE_COMMIT.txt').write_text(commit + '\n', encoding='utf-8')
    return src


def ablation_model(name, device):
    _, run_relative = RUNS[name]
    run = BASE / 'records' / run_relative
    manifest = read_json(run / 'manifest.json')
    cfg = manifest['config']
    src = ensure_ablation_source(name)
    arch, _ = runtime(src)
    for folder in ['linear_gate', 'gated_bypass', 'spatial_gate6']:
        sys.path.insert(0, str(src / 'research' / folder))
    if name == 'B0':
        model = arch.SPAN(**cfg['model'])
    elif name == 'SG6':
        model = load_module('paper_sg6', src / 'research/spatial_gate6/sg_model.py').build_model(arch, cfg)
    else:
        entry, _ = check_ablation_entry(name, src)
        sys.path.insert(0, str(entry.parent))
        module = load_module(f'paper_candidate_{name}', entry)
        model = module.build_model(arch, cfg)
    checkpoint = run / 'epoch00048_u48000.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['update'] != 48000:
        raise ValueError('Wrong ablation update')
    model.load_state_dict(state['model'], strict=True)
    provenance = dict(
        path=str(checkpoint),
        sha256=sha(checkpoint),
        weight_key='model',
        source_commit=RUNS[name][0],
        bound8=False,
    )
    if name in ABLATION_ENTRIES:
        entry, blob = check_ablation_entry(name, src)
        provenance.update(factory_path=str(entry), factory_git_blob=blob)
    return model.to(device).eval(), run, provenance


def _find_ngswin_deps():
    """Find isolated timm==0.6.13 + einops==0.3.0 directory created by step5, or install cleanly."""
    candidates = [
        ROOT / 'step5-ngswin-png-py311-fix01/deps',
        ROOT / 'step5-ngswin-png-py311-fix01/python_deps',
        ROOT / 'step5-ngswin-png-py311-fix01/site-packages',
        OUT / 'ngswin_deps',
    ]
    for p in ROOT.glob('step5-ngswin*/*'):
        if p.is_dir() and (p / 'einops').is_dir() and (p / 'timm').is_dir():
            candidates.insert(0, p)
    for c in candidates:
        if c.is_dir() and (c / 'einops').is_dir() and (c / 'timm').is_dir():
            return c
    target = OUT / 'ngswin_deps'
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(target),
         'timm==0.6.13', 'einops==0.3.0'],
        check=True,
    )
    return target


def public_model(name, device, deploy_fold=False):
    if name == 'SPAN52':
        arch, _ = runtime()
        model = arch.SPAN(num_in_ch=3, num_out_ch=3, feature_channels=52, upscale=4, img_range=255.0)
        path = ROOT / 'weights/4x-spanx4_ch52.pth'
        expected = 'db3aab7cbd91d47d747dd2320c8f915e30580ac87c59826869c909cd094a397f'
        if not path.exists():
            choices = [p for p in (ROOT / 'weights').glob('*span*.pth') if sha(p) == expected]
            if len(choices) != 1:
                raise FileNotFoundError('Cannot locate exact released SPAN52 weight')
            path = choices[0]
        if sha(path) != expected:
            raise ValueError('Wrong SPAN52 weight')
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False), strict=True)
        if deploy_fold:
            model = fused_copy(model, arch.Conv3XC)
        return model.to(device).eval(), dict(path=str(path), sha256=sha(path), mode='single',
                                             parameters=sum(p.numel() for p in model.parameters()))

    if name == 'DDistill-SR':
        folder = OUT / 'external/DDistill-SR'
        entries = []
        for rel, blob in [
            ('DDistill_SR_Inference_arch.py', 'b099e6c356795a6895c67b538fdb3625d97eabf1'),
            ('model-backup/DDistill-SRx4_inference.pth', '2a73d71e323b1ed0af10513ab3f728f9001a0f54'),
        ]:
            entries.append(fetch(f'https://raw.githubusercontent.com/icandle/DDistill-SR/{DD_COMMIT}/{rel}',
                                 folder / rel, blob=blob))
        module = load_module('paper_ddistill', folder / 'DDistill_SR_Inference_arch.py')
        model = module.DDistill_SR_Inference(
            in_nc=3, out_nc=3, nf=56, num_modules=4, scale=4, deploy=True, dynamic=True, L=16, style='DBB', res=True
        )
        ckpt = folder / 'model-backup/DDistill-SRx4_inference.pth'
        model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False), strict=True)
        return model.to(device).eval(), dict(repository='https://github.com/icandle/DDistill-SR', commit=DD_COMMIT,
                                             files=entries, parameters=sum(p.numel() for p in model.parameters()))

    if name == 'IMDN':
        folder = OUT / 'external/IMDN'
        entries = []
        for rel, blob in [
            ('model/architecture.py', 'b297f9c39bc4cd8398b18d9da705a693b3519c9c'),
            ('model/block.py', '86feb771614dacb83966d95989f9cc4c239cf60d'),
            ('checkpoints/IMDN_x4.pth', '3b9bf90db6985f7f97015a6629d001c46f3e2c33'),
        ]:
            entries.append(fetch(f'https://raw.githubusercontent.com/Zheng222/IMDN/{IMDN_COMMIT}/{rel}',
                                 folder / rel, blob=blob))
        pkg = types.ModuleType('paper_imdn_pkg')
        pkg.__path__ = [str(folder / 'model')]
        sys.modules[pkg.__name__] = pkg
        module = load_module('paper_imdn_pkg.architecture', folder / 'model/architecture.py')
        model = module.IMDN(upscale=4)
        state = torch.load(folder / 'checkpoints/IMDN_x4.pth', map_location='cpu', weights_only=False)
        if all(k.startswith('module.') for k in state):
            state = {k[len('module.'):]: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        return model.to(device).eval(), dict(repository='https://github.com/Zheng222/IMDN', commit=IMDN_COMMIT,
                                             files=entries, parameters=sum(p.numel() for p in model.parameters()))

    if name == 'LBNet':
        folder = OUT / 'external/LBNet'
        entries = []
        for rel, blob in [
            ('model/lbnet.py', 'ef8db748f406fb4d3c46a2252a1c69ced0fb2108'),
            ('model/common.py', '3d983e3bbb349ec50d345df0fa1fa419184ba369'),
            ('util/rlutrans.py', '79fd81e1b3f36c272cb17022b318dca66860b094'),
            ('util/tools.py', 'd905022ca27a41951e01d4c6d8fc28882a34b01f'),
            ('test_model/LBNet/LBNet-X4.pt', 'f1ded1c3c6aa33bdfa037efa02728125c9e15068'),
        ]:
            entries.append(fetch(f'https://raw.githubusercontent.com/wzx0826/LBNet/{LBNET_COMMIT}/{rel}',
                                 folder / rel, blob=blob))
        sys.path.insert(0, str(folder))
        from model.lbnet import LBNet
        args = types.SimpleNamespace(scale=[4], n_feats=32, num_heads=8, rgb_range=255, n_colors=3)
        core = LBNet(args)
        st = torch.load(folder / 'test_model/LBNet/LBNet-X4.pt', map_location='cpu', weights_only=False)
        own, got = set(core.state_dict().keys()), set(st.keys())
        missing, extra = own - got, got - own
        if missing:
            raise ValueError(f'LBNet missing keys: {missing}')
        if extra != {'norm.weight', 'norm.bias', 'proj.weight', 'proj.bias'}:
            raise ValueError(f'LBNet unexpected extra keys: {extra}')
        core.load_state_dict(st, strict=False)

        class Range255Wrapper(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, x):
                return self.net(x * 255.0) / 255.0

        model = Range255Wrapper(core)
        return model.to(device).eval(), dict(
            repository='https://github.com/wzx0826/LBNet',
            commit=LBNET_COMMIT,
            files=entries,
            ignored_unused_official_keys=sorted(extra),
            parameters=sum(p.numel() for p in core.parameters()),
        )

    if name in ('SMFANet', 'SMFANet+'):
        folder = OUT / 'external/SMFANet'
        is_plus = (name == 'SMFANet+')
        w_rel = 'pretrain/SMFANet_plus_DF2K_100w_x4SR.pth' if is_plus else 'pretrain/SMFANet_DF2K_100w_x4SR.pth'
        w_blob = '07408ad45812dc16ecdc38e37d4dfbced5ecf98b' if is_plus else 'ad046feae3ae874490669548bde106cd62dcf752'
        entries = [
            fetch(f'https://raw.githubusercontent.com/Zheng-MJ/SMFANet/{SMFA_COMMIT}/basicsr/archs/SMFANet_arch.py',
                  folder / 'basicsr/archs/SMFANet_arch.py', blob='cd058673056c605d5ca4ee5855e59f810b1233b8'),
            fetch(f'https://raw.githubusercontent.com/Zheng-MJ/SMFANet/{SMFA_COMMIT}/{w_rel}',
                  folder / w_rel, blob=w_blob),
        ]
        # Provide lightweight stub for basicsr.utils.registry if not already loaded.
        if 'basicsr.utils.registry' not in sys.modules:
            sys.modules.setdefault('basicsr', types.ModuleType('basicsr'))
            sys.modules.setdefault('basicsr.utils', types.ModuleType('basicsr.utils'))
            mreg = types.ModuleType('basicsr.utils.registry')
            mreg.ARCH_REGISTRY = types.SimpleNamespace(register=lambda: (lambda cls: cls))
            sys.modules['basicsr.utils.registry'] = mreg
        mod = load_module('paper_smfanet', folder / 'basicsr/archs/SMFANet_arch.py')
        dim, n_blocks = (48, 12) if is_plus else (36, 8)
        model = mod.SMFANet(dim=dim, n_blocks=n_blocks, ffn_scale=2, upscaling_factor=4)
        st = torch.load(folder / w_rel, map_location='cpu', weights_only=False)
        sd = st['params'] if isinstance(st, dict) and 'params' in st else st
        model.load_state_dict(sd, strict=True)
        return model.to(device).eval(), dict(
            repository='https://github.com/Zheng-MJ/SMFANet',
            commit=SMFA_COMMIT,
            files=entries,
            config=dict(dim=dim, n_blocks=n_blocks, ffn_scale=2, upscaling_factor=4),
            parameters=sum(p.numel() for p in model.parameters()),
        )

    if name == 'SwinIR-light':
        sys.path.insert(0, str(ROOT / 'python_deps'))
        mod = load_module('paper_swinir', ROOT / 'external/SwinIR/models/network_swinir.py')
        core = mod.SwinIR(
            upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0,
            depths=[6] * 4, embed_dim=60, num_heads=[6] * 4, mlp_ratio=2,
            upsampler='pixelshuffledirect', resi_connection='1conv',
        )
        path = ROOT / 'weights/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth'
        if sha(path) != '09fad24e32ae62722e1a055efde9921328f4137981bab0a42a4a3a806306c58e':
            raise ValueError('Wrong SwinIR weight')
        core.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['params'], strict=True)

        class Padded(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.core = core

            def forward(self, x):
                h, w = x.shape[-2:]
                ph = (h // 8 + 1) * 8 - h
                pw = (w // 8 + 1) * 8 - w
                padded = torch.cat([x, x.flip([2])], 2)[:, :, :h + ph, :]
                padded = torch.cat([padded, padded.flip([3])], 3)[:, :, :, :w + pw]
                return self.core(padded)[:, :, :h * 4, :w * 4]

        model = Padded()
        return model.to(device).eval(), dict(path=str(path), sha256=sha(path), mode='single',
                                             parameters=sum(p.numel() for p in core.parameters()))

    if name == 'NGswin':
        deps = _find_ngswin_deps()
        ng_root = ROOT / 'external/NGramSwin'
        for p in (str(deps), str(ng_root)):
            if p not in sys.path:
                sys.path.insert(0, p)
        from my_model.ngswin_model.ngswin import NGswin
        core = NGswin(target_mode='light_x4')
        path = ng_root / 'pretrain/NGswin_x4.pth'
        expected = '13766fa818c8dd557a582f25408169fc1986aa0bf7274a1c61a8ca7c1e427ec7'
        if sha(path) != expected:
            raise ValueError(f'NGswin weight hash mismatch: {path}')
        raw = torch.load(path, map_location='cpu', weights_only=False)
        sd = raw['model'] if isinstance(raw, dict) and 'model' in raw else raw
        sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
        own = core.state_dict()
        buf_keys = {k for k, _ in core.named_buffers()}
        param_keys = {k for k, _ in core.named_parameters()}
        missing = set(own.keys()) - set(sd.keys())
        missing_params = missing & param_keys
        if missing_params:
            raise ValueError(f'NGswin missing learnable parameters: {missing_params}')
        for k in missing:
            if k not in buf_keys:
                raise ValueError(f'NGswin unexpected missing key: {k}')
            sd[k] = own[k].clone()
        core.load_state_dict(sd, strict=True)

        class NGswinPadded(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, x):
                _, _, h, w = x.shape
                padh = (32 - h % 32) if (h % 32) != 0 else 0
                padw = (32 - w % 32) if (w % 32) != 0 else 0
                if padh > 0:
                    x = torch.cat([x, x.flip([2])], dim=2)[:, :, :h + padh, :]
                if padw > 0:
                    x = torch.cat([x, x.flip([3])], dim=3)[:, :, :, :w + padw]
                return self.net(x)[:, :, :h * 4, :w * 4]

        model = NGswinPadded(core)
        return model.to(device).eval(), dict(
            path=str(path),
            sha256=sha(path),
            reconstructed_deterministic_buffers=len(missing),
            parameters=sum(p.numel() for p in core.parameters()),
        )

    raise ValueError(f'Unknown public model: {name}')
