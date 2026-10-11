import json
from train import pairs, aligned, digest
from runtime import ROOT

def main():
    paths=json.loads((ROOT/'research/configs/paths.local.json').read_text(encoding='utf-8-sig'))
    sets={'DIV2K_train':paths['train'][0],**paths['validation']}; report={}; hashes={}
    for name,spec in sets.items():
        data=pairs(spec); shapes=[]; hashes[name]=set()
        for hp,lp in data:
            hr,lr=aligned(hp,lp); shapes.append(list(lr.shape[:2])); hashes[name].add(digest(hp))
        report[name]=dict(count=len(data),pair_shapes_checked=True,min_lr_hw=[min(s[i] for s in shapes) for i in (0,1)],max_lr_hw=[max(s[i] for s in shapes) for i in (0,1)])
        print(name,report[name],flush=True)
    for name in paths['validation']:
        if hashes['DIV2K_train'] & hashes[name]: raise ValueError(f'Leakage: {name}')
    report['provenance']='User-provided existing data; bicubic kernel provenance not independently verified. Flickr2K pending.'
    target=ROOT/'research/docs/data-audit.json'; target.write_text(json.dumps(report,indent=2),encoding='utf8')
if __name__=='__main__': main()
