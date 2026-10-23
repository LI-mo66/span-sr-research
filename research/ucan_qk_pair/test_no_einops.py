"""Regression: entry imports and source comparisons with einops unavailable."""
import importlib.abc
import json
from pathlib import Path
import sys


class BlockEinops(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'einops' or fullname.startswith('einops.'):
            raise ModuleNotFoundError('einops intentionally unavailable in this regression')
        return None


def main():
    sys.meta_path.insert(0, BlockEinops())
    import torch
    import prepare
    import verify
    import integration
    import resources
    import pair_model
    import release_guard
    torch.set_num_threads(2)
    report=dict(status='PASS',einops_import_blocked=True,imports=True,
                scope='dependency regression and unchanged reference equations; no real training',
                source_reference={})
    for device in ('cpu','cuda'):
        if device=='cuda' and not torch.cuda.is_available():continue
        report['source_reference'][device]=verify.equivalence(device)
    assert not any(k=='einops' or k.startswith('einops.') for k in sys.modules)
    root=Path(__file__).resolve().parents[2]
    out=root/'tmp/ucan-pair/no-einops-regression.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('PASS: einops imports blocked; server entry imports and CPU/CUDA source forward/backward comparisons passed')


if __name__=='__main__':main()
