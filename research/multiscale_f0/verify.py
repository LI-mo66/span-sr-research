"""Full-model shape/oracle checks and real-entry exact continuation tests."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,json,random,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(1,str(ROOT/'research'))
import fusion
import trainer
from ensemble import predict,transform,TRANSFORMS

def equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)

def main():
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    folder=ROOT/'logs'/('multiscale-verify-'+str(time.time_ns()));folder.mkdir(parents=True)
    arch,_=fusion.components();template=json.loads((HERE/'template.json').read_text())
    reports=[]
    for scale in (2,3):
        cfg=copy.deepcopy(template);cfg['model']['upscale']=scale
        trainer.SCALE=scale
        class Directional(torch.nn.Module):
            upscale=scale
            def forward(self,x):
                y=x.repeat_interleave(scale,-2).repeat_interleave(scale,-1)
                return y+torch.arange(y.shape[-1],device=y.device)[None,None,None,:]*.1
        probe=torch.arange(3*3*5,dtype=torch.float32).reshape(1,3,3,5)
        plain,average=predict(Directional(),probe)
        oracle=[]
        for flip in (False,True):
            for rotation in range(4):
                x=probe.flip(-1) if flip else probe
                x=torch.rot90(x,rotation,(-2,-1))
                y=Directional()(x)
                y=torch.rot90(y,-rotation,(-2,-1))
                oracle.append(y.flip(-1) if flip else y)
        torch.testing.assert_close(average,torch.stack(oracle).mean(0),atol=2e-5,rtol=1e-6)
        torch.testing.assert_close(plain,Directional()(probe),atol=0,rtol=0)
        # Exact crop geometry incl grayscale and non-divisible HR border.
        low=np.arange(20*23,dtype=np.uint16).reshape(20,23)%251
        low=low.astype(np.uint8);high=np.repeat(np.repeat(low,scale,0),scale,1)
        high=np.pad(high,((0,scale-1),(0,scale-1)),mode='edge')
        hp=folder/f'h{scale}.png';lp=folder/f'l{scale}.png'
        cv2.imwrite(str(hp),high);cv2.imwrite(str(lp),low)
        x,y=trainer.sample((hp,lp),8*scale,random.Random(7))
        assert x.shape==(3,8,8) and y.shape==(3,8*scale,8*scale)
        assert torch.equal(y,x.repeat_interleave(scale,-2).repeat_interleave(scale,-1))
        for arm in ('B0','F0'):
            if os.environ.get('MS_VERIFY_CASE') and os.environ['MS_VERIFY_CASE']!=f'{scale}:{arm}':continue
            cfg['arm']=arm
            for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                torch.manual_seed(10);model=fusion.build_model(arch,cfg).to(device)
                if arm=='B0':
                    torch.manual_seed(10);oracle=arch.SPAN(**cfg['model']).to(device)
                    equal(model.state_dict(),oracle.state_dict());del oracle
                else:assert model.image_residual.out_channels==3*scale*scale
                for h,w in ((1,1),(3,5),(17,19)):
                    x=torch.rand(1,3,h,w,device=device)
                    with torch.no_grad():plain,average=predict(model,x)
                    assert plain.shape==(1,3,h*scale,w*scale)
                    assert average.shape==plain.shape
                    for code in TRANSFORMS:
                        assert torch.equal(x,transform(transform(x,code),code,True))
                opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=2e-4)
                x=torch.rand(1,3,8,8,device=device)
                for _ in range(3):
                    opt.zero_grad(set_to_none=True);loss=model(x).abs().mean();loss.backward()
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad)
                    opt.step()
                print('PASS model',scale,arm,device,flush=True)
                del model,opt
            if not torch.cuda.is_available():continue
            # Real trainer fixture: all datasets, four updates, one milestone crossed.
            paths={'train':None,'validation':{}}
            for i,name in enumerate(('train','DIV2K_valid','Set5','Set14','B100','Urban100','Manga109')):
                d=folder/f'{scale}-{arm}-{name}';(d/'HR').mkdir(parents=True);(d/'LR').mkdir()
                rng=np.random.default_rng(i+123)
                image=rng.integers(0,256,(20,23,3),dtype=np.uint8)
                cv2.imwrite(str(d/'LR/a.png'),image)
                cv2.imwrite(str(d/'HR/a.png'),np.repeat(np.repeat(image,scale,0),scale,1))
                spec=dict(hr=str(d/'HR'),lr=str(d/'LR'))
                if name=='train':paths['train']=spec
                else:paths['validation'][name]=spec
            pf=folder/f'paths{scale}-{arm}.json';pf.write_text(json.dumps(paths))
            ecfg=dict(cfg,engineering_only=True,total_updates=4,record_updates=2,
                effective_batch=2,micro_batch=1,hr_patch=8*scale,milestones=[2,3],
                image_backend='png',paths_file=str(pf),crop_border=scale)
            cf=folder/f'{scale}-{arm}.json';cf.write_text(json.dumps(ecfg))
            stem=f'MS-VERIFY-{scale}-{arm}-{folder.name.split("-")[-1]}'
            def invoke(name,cap,resume=False):
                command=[sys.executable,str(HERE/'run.py'),'--config',str(cf),'--id',stem+name,'--until-update',str(cap)]
                if resume:command.append('--resume')
                with (folder/f'{scale}-{arm}-{name}-{cap}.log').open('w') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            invoke('staged',2);invoke('staged',4,True);invoke('direct',4)
            a=torch.load(ROOT/'runs'/(stem+'staged')/'latest.pt',map_location='cpu',weights_only=False)
            b=torch.load(ROOT/'runs'/(stem+'direct')/'latest.pt',map_location='cpu',weights_only=False)
            for key in ('model','optimizer','scheduler','python_rng','numpy_rng','torch_rng','cuda_rng'):
                equal(a[key],b[key])
            assert a['scheduler']['_last_lr']==[5e-5]
            for dataset in paths['validation']:
                assert dataset in a['epoch_record']['metrics'] and dataset+'_x8' in a['epoch_record']['metrics']
            reports.append(dict(scale=scale,arm=arm,exact_resume=True,metrics=True))
            print('PASS trainer exact resume',scale,arm,flush=True)
    result=dict(status='PASS',torch=torch.__version__,device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',tests=reports,scope='Engineering checks only; no accuracy guarantee')
    (folder/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
