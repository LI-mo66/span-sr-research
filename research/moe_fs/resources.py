"""Reuse identical resource protocol with the isolated FusionGate4 factory."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fs_runtime', HERE / 'runtime.py')
fs_runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs_runtime)
from model import build_model

if __name__ == '__main__':
    profile = fs_runtime.load_shared('moe_fs_shared_profile', 'profile_candidate.py')
    profile.build_model = build_model
    profile.main()
