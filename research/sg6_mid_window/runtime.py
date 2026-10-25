"""Use fixed shared trainer and metrics unchanged."""
import importlib.util
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent
sys.path.insert(0, str(RESEARCH))


def load_shared(name, filename):
    spec = importlib.util.spec_from_file_location(name, RESEARCH / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


shared_runtime = load_shared('runtime', 'runtime.py')
components = shared_runtime.components
trainer = load_shared('mid_shared_trainer', 'moe_fs/trainer.py')
# Keep this experiment's prepare/run/model imports ahead of legacy directories.
# Shared trainer already holds its explicitly loaded runtime/components bindings.
sys.path.insert(0, str(Path(__file__).resolve().parent))
