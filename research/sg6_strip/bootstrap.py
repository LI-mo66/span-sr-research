"""Load unchanged shared SG6 trainer without importing unrelated candidate factories."""
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RESEARCH=ROOT/'research'
sys.path.insert(0,str(RESEARCH))

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

shared=load('runtime',RESEARCH/'runtime.py')
components=shared.components
trainer=load('strip_shared_trainer',RESEARCH/'moe_fs/trainer.py')
for rel in ('linear_gate','gated_bypass','spatial_gate6'):
    sys.path.insert(0,str(RESEARCH/rel))
