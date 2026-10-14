"""Import the existing shared harness under unambiguous module names."""
import importlib.util
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent
# Shared trainer imports runtime by name; load shared runtime explicitly and
# make it available before loading trainer. This file is imported as easa_runtime.
sys.path.insert(0, str(RESEARCH))


def load_shared(name, filename):
    spec = importlib.util.spec_from_file_location(name, RESEARCH / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


shared_runtime = load_shared('runtime', 'runtime.py')
components = shared_runtime.components
trainer = load_shared('easa_shared_trainer', 'moe_fs/trainer.py')
