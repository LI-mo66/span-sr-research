"""Reproduce archived F0 plain metrics, measure x8, test largest real image."""
import importlib.util
import json
import sys
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research/fusion_wu'))
import fusion
sys.path.insert(0,str(HERE))
import torch
from ensemble import predict

def main():
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    base=Path('/home/featurize/work/span-x4-research/records/FEAT-SG6-W-UCAN-P24-LKSA5-FUSION-V1-50K-01-1793142163941762977')
    assert (base/'BACKUP_COMPLETE').is_file()
    oldrun=base/'runs/FEAT-SG6-W-UCAN-P24-LKSA5-FUSION-V1-50K-01'
    history=[json.loads(s) for s in (oldrun/'epochs.jsonl').read_text().splitlines() if s.strip()]
    old=min(history,key=lambda r:(-r['metrics']['DIV2K_valid']['psnr'],r['update']))
    assert old['update']==48000
    checkpoint=oldrun/'epoch00048_u48000.pt';before=fusion.trainer.digest(checkpoint)
    manifest=json.loads((base/'backup-manifest.json').read_text())
    match=next(r for r in manifest['files'] if r['path']==str(checkpoint.relative_to(base)))
    assert before==match['sha256']
    cfg=json.loads((ROOT/'logs/f0-long/frozen/config.json').read_text())
    paths=json.loads(Path(cfg['paths_file']).read_text())
    data={k:fusion.trainer.pairs(v) for k,v in paths['validation'].items()}
    spec=importlib.util.spec_from_file_location('check_long_trainer',HERE/'trainer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    arch,metrics=fusion.components();model=fusion.build_model(arch,cfg)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True);del state
    model=model.cuda();model.requires_grad_(False)
    destination=ROOT/'logs/f0-long/reproduction-48K';destination.mkdir()
    start=time.perf_counter()
    summary=module.evaluate(model,data,metrics,cfg,destination,48,48000)
    elapsed=time.perf_counter()-start
    for name,values in old['metrics'].items():
        for metric in ('psnr','ssim'):
            assert abs(summary[name][metric]-values[metric])<1e-8,(name,metric,summary[name],values)
    model.eval();largest=None;area=0
    for pairs in data.values():
        for hp,lp in pairs:
            image=fusion.trainer.read(lp);size=image.shape[0]*image.shape[1]
            if size>area:largest=lp;area=size
    image=fusion.trainer.tensor(fusion.trainer.read(largest)).unsqueeze(0).cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():plain,average=predict(model,image)
    torch.cuda.synchronize()
    assert torch.isfinite(average).all() and average.shape==plain.shape
    assert fusion.trainer.digest(checkpoint)==before
    report=dict(status='PASS',plain_metrics_reproduced=True,checkpoint_sha256=before,
        selected_48K_plain_and_x8=summary,six_dataset_dual_eval_seconds=elapsed,
        largest_lr=str(largest),largest_input_shape=list(image.shape),
        largest_x8_peak_allocated=torch.cuda.max_memory_allocated(),
        scope='Archived48K weights used for testing only; formal model independently initialized from seed10')
    (ROOT/'logs/f0-long/server-check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
