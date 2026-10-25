"""Compare archived and candidate numerical sensitivity without changing training."""
import json
import torch
from verify import ROOT, baseline, build_model, load_arch

torch.set_num_threads(2)
torch.backends.cudnn.benchmark=False
torch.use_deterministic_algorithms(True)
arch=load_arch()
cfg=json.loads((ROOT/'research/spatial_gate6/configs/LG6-SiLU-SG6.json').read_text())
records=[]
for tf32 in (True,False):
    torch.backends.cudnn.allow_tf32=tf32
    torch.backends.cuda.matmul.allow_tf32=False
    for sign in (0,1,-1):
        torch.manual_seed(10)
        model=(baseline(arch,cfg) if sign==0 else build_model(
            arch,dict(cfg,feature_evolution=dict(block=3,width=13,sign=sign)))).cuda()
        torch.manual_seed(91)
        for shape in ((1,3,1,1),(1,3,13,17)):
            x=torch.rand(*shape,device='cuda')
            with torch.no_grad():
                model.train();a=model(x)
                model.eval();b=model(x)
            records.append(dict(cudnn_tf32=tf32,sign=sign,shape=shape,
                                train_eval_absmax=float((a-b).abs().max())))
output=ROOT/'research/feature_evolution/precision-probe.json'
output.write_text(json.dumps(dict(evidence='synthetic numerical diagnostic only',
    device=torch.cuda.get_device_name(),torch=torch.__version__,records=records),indent=2),encoding='utf-8')
print(output)
