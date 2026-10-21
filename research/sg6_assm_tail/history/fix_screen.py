import copy,json,sys,math
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path('research/sg6_assm').resolve()))
from model import build_model,build_sg6,BlockWithASSM
from adapter import ASSMAdapter
import runtime

def rms(x):return float(x.detach().square().mean().sqrt())
class RMSAdapter(ASSMAdapter):
 def forward(self,image):
  b,c,h,w=image.shape
  value=self.core(self.norm(image.flatten(2).transpose(1,2)),(h,w))*self.scale
  amplitude=image.detach().square().mean(1,keepdim=True).clamp_min(1e-12).sqrt()
  return image+amplitude*value.transpose(1,2).reshape(b,c,h,w)

torch.set_num_threads(2)
cfg=json.loads(Path('research/sg6_assm/configs/M1-mid3-lr2e4.json').read_text());cfg.update(engineering_only=True);cfg['assm_settings']['backend']='reference'
arch,_=runtime.components();torch.manual_seed(10);base=build_sg6(arch,cfg);torch.manual_seed(10);proto=build_model(arch,cfg)
x=torch.rand(2,3,13,17);target=torch.nn.functional.interpolate(x,scale_factor=4,mode='bicubic',align_corners=False).clamp(0,1)
with torch.no_grad():baseline=base(x)
rows=[]
for name,position,alpha,prompt_scaled,rms_match in [('mid_old','mid',.01,False,False),('tail_a01_old','tail',.1,False,False),('tail_a01_prompt','tail',.1,True,False),('tail_a01_prompt_rms','tail',.1,True,True),('tail_a001_prompt_rms','tail',.01,True,True),('mid_a01_prompt_rms','mid',.1,True,True)]:
 m=copy.deepcopy(base);a=copy.deepcopy(proto.block_3.adapter)
 with torch.no_grad():
  a.scale.fill_(alpha)
  if prompt_scaled:
   a.core.embeddingA.weight.mul_(math.sqrt(3*32))
   a.core.embeddingB.weight.mul_(math.sqrt(3*64))
 if rms_match:a.__class__=RMSAdapter
 if position=='mid':m.block_3=BlockWithASSM(m.block_3,a)
 else:m.conv_cat=nn.Sequential(m.conv_cat,a)
 cap={}
 h1=a.register_forward_pre_hook(lambda mo,i:cap.update(F=i[0].detach().clone()))
 h2=a.register_forward_hook(lambda mo,i,o:cap.update(Y=o.detach().clone()))
 y=m(x);h1.remove();h2.remove()
 loss=torch.nn.functional.l1_loss(y,target);loss.backward()
 before={k:p.detach().clone() for k,p in a.named_parameters()};grads={k:rms(p.grad) for k,p in a.named_parameters()}
 opt=torch.optim.Adam(m.parameters(),lr=2e-4,betas=(.9,.999),eps=1e-8);opt.step()
 row=dict(name=name,loss=float(loss.detach()),ratio=rms(cap['Y']-cap['F'])/rms(cap['F']),rgb_change=rms(y-baseline),updates={k:dict(grad=grads[k],relative=rms(p-before[k])/max(rms(before[k]),1e-12)) for k,p in a.named_parameters()})
 rows.append(row)
 print(name,'ratio',row['ratio'],'route',row['updates']['core.route.0.weight'],'state',row['updates']['core.selectiveScan.A_logs'])
Path('tmp/assm_fix_screen.json').write_text(json.dumps(rows,indent=2))
