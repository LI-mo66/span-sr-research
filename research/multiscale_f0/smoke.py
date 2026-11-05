import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,random,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(1,str(ROOT/'research'))
import fusion,trainer
from ensemble import predict

cfg=json.loads(Path(sys.argv[1]).read_text());trainer.SCALE=cfg['model']['upscale']
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=True
torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
random.seed(10);np.random.seed(10);torch.manual_seed(10)
paths=json.loads(Path(cfg['paths_file']).read_text());train=trainer.pairs(paths['train'])
arch,_=fusion.components();model=fusion.build_model(arch,cfg).cuda()
opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=2e-4)
torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True)
for i in range(4):
    samples=[trainer.sample(random.choice(train),cfg['hr_patch']) for _ in range(16)]
    low=torch.stack([s[0] for s in samples]).cuda();high=torch.stack([s[1] for s in samples]).cuda()
    prediction=model(low);assert prediction.shape==high.shape
    loss=torch.nn.functional.l1_loss(prediction,high)/4
    assert bool(torch.isfinite(loss));loss.backward()
assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad)
opt.step();training_peak=torch.cuda.max_memory_allocated()
del opt,loss,prediction,low,high,samples
model.eval();largest=None;area=0
for spec in paths['validation'].values():
    for hp,lp in trainer.pairs(spec):
        from PIL import Image
        with Image.open(lp) as im:w,h=im.size
        if w*h>area:area=w*h;largest=lp
with torch.inference_mode():
    x=trainer.tensor(trainer.read(largest)).unsqueeze(0).cuda()
    torch.cuda.reset_peak_memory_stats();plain,average=predict(model,x)
    assert torch.isfinite(average).all()
result=dict(status='PASS',scale=cfg['model']['upscale'],arm=cfg['arm'],micro_batch=16,accumulation=4,training_peak_bytes=training_peak,
    largest_eval=str(largest),eval_shape=list(x.shape),output_shape=list(average.shape),eval_peak_bytes=torch.cuda.max_memory_allocated())
from deploy import fused_copy
result['deployed_parameters']=sum(p.numel() for p in fused_copy(model,arch.Conv3XC).parameters())
Path(sys.argv[2]).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
