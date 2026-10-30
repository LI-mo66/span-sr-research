"""SG6 + unchanged W/U branches, post-main-output wrappers only."""
import importlib.util
import sys
from pathlib import Path
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for folder in ('spatial_gate6', 'gated_bypass', 'linear_gate'):
    sys.path.insert(0, str(ROOT/'research'/folder))
sys.path.insert(0, str(ROOT/'research'))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

shared = load('runtime', ROOT/'research/runtime.py')
trainer = load('fusion_shared_trainer', ROOT/'research/moe_fs/trainer.py')
components = shared.components
from sg_model import build_model as build_sg6
attention = load('fusion_context', HERE/'attention.py')
window = load('fusion_window', HERE/'window_adapter.py')
lksa = load('fusion_lksa', HERE/'lksa.py')
sys.path.insert(0, str(HERE))

class LKSAAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = lksa.LKSA(52, k=35, ratio=4)
        self.project = nn.Conv2d(52,52,1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self,x):
        scale = x.detach().square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-6)
        return x + scale*self.project(self.core(x/scale))

class PostBlock(nn.Module):
    def __init__(self, original, adapter, context=False):
        super().__init__()
        self.original, self.adapter, self.context = original, adapter, context

    def forward(self,x):
        main,first,gate = self.original(x)
        value = self.adapter(main)
        if self.context:
            value = value[0]
        return value,first,gate

def validate(cfg):
    expected = dict(positions=[2,3,4,5],window=16,window_channels=26,
                    window_heads=2,window_initial_scale=.03,lksa_kernel=35,
                    context='independent',initialization='unchanged-W-U')
    if cfg.get('fusion_settings') != expected:
        raise ValueError('Unregistered fusion configuration')
    if cfg.get('experiment_version') != 'SG6-W-UCAN-P24-LKSA5-FUSION-v1':
        raise ValueError('Unknown experiment version')

def build_model(arch,cfg):
    validate(cfg)
    model = build_sg6(arch,cfg)
    devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg['seed']+80001)
        contexts = [attention.ContextAdapter(), attention.ContextAdapter()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg['seed']+90001)
        spatial = LKSAAdapter()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg['seed']+70001)
        interaction = window.PartialInteraction('A',.03,128)
    for position,adapter,context in ((2,contexts[0],True),(3,interaction,False),
                                     (4,contexts[1],True),(5,spatial,False)):
        name = f'block_{position}'
        setattr(model,name,PostBlock(getattr(model,name),adapter,context))
    model.extra_parameters += 34123
    assert model.extra_parameters == 71687
    return model

def original_backbone_state(model):
    return {k.replace('.original.','.') : v for k,v in model.state_dict().items()
            if '.adapter.' not in k}
