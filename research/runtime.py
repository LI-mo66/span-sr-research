"""Load pinned upstream components without executing unrelated BasicSR plugins."""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def components():
    for name in ('basicsr', 'basicsr.utils', 'basicsr.metrics', 'basicsr.archs'):
        module = types.ModuleType(name)
        module.__path__ = [str(ROOT / name.replace('.', '/'))]
        sys.modules[name] = module
    load('basicsr.utils.registry', 'basicsr/utils/registry.py')
    color = load('basicsr.utils.color_util', 'basicsr/utils/color_util.py')
    sys.modules['basicsr.utils'].bgr2ycbcr = color.bgr2ycbcr
    load('basicsr.metrics.metric_util', 'basicsr/metrics/metric_util.py')
    metrics = load('basicsr.metrics.psnr_ssim', 'basicsr/metrics/psnr_ssim.py')
    arch = load('basicsr.archs.span_arch', 'basicsr/archs/span_arch.py')
    return arch, metrics
