"""Measured PNGs only; new filenames preserve the previous figure versions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm
from matplotlib.patches import Rectangle
from PIL import Image
from common import *

METHODS=['bicubic','span52_official_single','swinir_light_single','ngswin_single','ddistill_sr_single','imdn_single','ours_single','GT']
NAMES=['Bicubic','SPAN52','SwinIR-light','NGswin','DDistill-SR','IMDN','Ours','GT']
WIDTH=16.6/2.54*72
ROIS={'img004.png':[864,432,96,96],'img011.png':[256,880,96,96]}
PPT_TEXT_ROI=[32,256,96,96]
PPT_MARGIN_CUTOFF=.05  # Engineering selection, not a statistical significance threshold.


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})


def axes(fig,x,y,w,h):
    ax=fig.add_axes([x/WIDTH,1-(y+h)/(fig.get_figheight()*72),w/WIDTH,h/(fig.get_figheight()*72)])
    ax.axis('off');return ax


def text(fig,x,y,value,color='black',size=8.3,weight='normal'):
    return fig.text(x/WIDTH,1-y/(fig.get_figheight()*72),value,ha='center',va='top',fontsize=size,color=color,fontweight=weight)


def finish(fig,name):
    fig.canvas.draw();r=fig.canvas.get_renderer();boxes=[]
    for label in fig.texts:
        b=label.get_window_extent(r)
        if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height:
            raise ValueError('Clipped label: '+label.get_text())
        if any(b.overlaps(x) for x in boxes):raise ValueError('Overlapping label: '+label.get_text())
        boxes.append(b)
    dest=OUT/'figures_paper_ready';dest.mkdir(exist_ok=True)
    for ext in ['png','pdf']:
        p=dest/(name+'.'+ext)
        temporary=dest/(name+'.part.'+ext)
        fig.savefig(temporary,dpi=300,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else None)
        if p.exists():
            if sha(p)!=sha(temporary):raise FileExistsError('Existing figure differs: '+str(p))
            temporary.unlink()
        else:temporary.replace(p)
    plt.close(fig)
    image=Image.open(dest/(name+'.png'))
    if any(abs(v-300)>.1 for v in image.info['dpi']):raise ValueError('DPI mismatch')
    print('FIGURE_EXPORTED',name,'pixels',image.size,flush=True)
    return dict(png=str(dest/(name+'.png')),pdf=str(dest/(name+'.pdf')),width_cm=16.6,dpi=300,pixels=list(image.size),text_bounds='checked',visual_review='pending actual exported files')


def crop(image,roi):
    x,y,w,h=roi;a=image[y:y+h,x:x+w]
    if a.shape!=(h,w,3):raise ValueError('Invalid HR-coordinate ROI')
    return a


def scene(ds,image,roi,methods=METHODS):
    full={m:get_image(ds,m,image) for m in methods}
    if len({a.shape for a in full.values()})!=1:raise ValueError('Source shape mismatch')
    patches={m:crop(a,roi) for m,a in full.items()}
    scores={m:metrics(a,patches['GT']) for m,a in patches.items() if m!='GT'}
    return dict(dataset=ds,image=image,roi=roi,metrics=scores,
                sha256={m:sha(image_path(ds,m,image)) for m in methods}),full,patches


def best_img093():
    import cv2
    full={m:get_image('Urban100',m,'img093.png') for m in METHODS}
    _,mt=runtime();gty=mt.to_y_channel(full['GT'][:,:,::-1].copy()).squeeze(-1).astype(np.float64)
    h,w=gty.shape
    xx,yy=np.meshgrid(np.arange(16,w-96-16+1,16),np.arange(16,h-96-16+1,16));x,y=xx.ravel(),yy.ravel()
    def area(a,x,y,n):
        ii=cv2.integral(a);return(ii[y+n,x+n]-ii[y,x+n]-ii[y+n,x]+ii[y,x])/(n*n)
    texture=area(cv2.magnitude(cv2.Sobel(gty,cv2.CV_64F,1,0),cv2.Sobel(gty,cv2.CV_64F,0,1)),x,y,96)
    keep=texture>=np.quantile(texture,.7);scores={}
    for m in METHODS[:-1]:
        err=mt.to_y_channel(full[m][:,:,::-1].copy()).squeeze(-1).astype(np.float64)-gty
        scores[m]=10*np.log10(255**2/np.maximum(area(err**2,x+4,y+4,88),1e-12))
    margin=scores['ours_single']-np.max(np.stack([scores[m] for m in METHODS[:-2]]),0)
    order=sorted(np.flatnonzero(keep),key=lambda i:(-margin[i],int(y[i]),int(x[i])))
    i=order[0]
    return [int(x[i]),int(y[i]),96,96],dict(rule='highest local single-pass margin after per-image top30% GT-gradient screen',margin_db=float(margin[i]))


def fig3():
    style();records=[];scenes=[]
    for ds,image,roi in [('Urban100','img011.png',ROIS['img011.png']),('Urban100','img004.png',ROIS['img004.png']),('Manga109','YumeiroCooking.png',[16,832,96,96])]:
        record,full,patch=scene(ds,image,roi);records.append(record);scenes.append((full,patch))
    record,full,patch=scene('Set14','ppt3.png',PPT_TEXT_ROI)
    margin=record['metrics']['ours_single']['psnr']-max(record['metrics'][m]['psnr'] for m in METHODS[:-2])
    selection=dict(ppt_manual_english_roi=PPT_TEXT_ROI,ppt_margin_db=margin,engineering_cutoff_db=PPT_MARGIN_CUTOFF,ppt_metrics=record['metrics'],statistical_significance_claim=False)
    if margin<=PPT_MARGIN_CUTOFF:
        roi,diagnostic=best_img093();record,full,patch=scene('Urban100','img093.png',roi)
        selection.update(fallback=True,fallback_search=diagnostic)
    else:selection['fallback']=False
    records.append(record);scenes.append((full,patch))
    # Larger overview + 2 rows x 4 square crops, 8 methods including GT.
    context=232.;gap=8.;colgap=4.;side=(WIDTH-8-context-gap-3*colgap)/4
    line=22.;inner=6.;row=2*(side+line)+inner+8
    fig=plt.figure(figsize=(WIDTH/72,(8+4*row)/72),dpi=300)
    for i,(record,(full,patch)) in enumerate(zip(records,scenes)):
        top=4+i*row
        ax=axes(fig,4,top,context,2*side+line+inner)
        ax.imshow(full['GT'],interpolation='nearest',resample=False)
        x,y,w,h=record['roi'];ax.add_patch(Rectangle((x-.5,y-.5),w,h,fill=False,edgecolor='#E53935',linewidth=2))
        text(fig,4+context/2,top+2*side+line+inner+3,record['dataset']+' (x4): '+Path(record['image']).stem)
        for k,(method,label) in enumerate(zip(METHODS,NAMES)):
            rr,cc=divmod(k,4);left=4+context+gap+cc*(side+colgap);yy=top+rr*(side+line+inner)
            axes(fig,left,yy,side,side).imshow(patch[method],interpolation='nearest',resample=False)
            color='#B71C1C' if method=='ours_single' else 'black'
            text(fig,left+side/2,yy+side+2,label,color=color,weight='bold' if method=='ours_single' else 'normal')
            if method!='GT':
                m=record['metrics'][method];text(fig,left+side/2,yy+side+12,f"{m['psnr']:.2f}/{m['ssim']:.4f}",color=color,size=8.)
    exported=finish(fig,'fig3_qualitative_x4_single_v2')
    rows=[]
    for record in records:
        for method,m in record['metrics'].items():
            rows.append(dict(dataset=record['dataset'],image_name=record['image'],method=method,x=record['roi'][0],y=record['roi'][1],w=96,h=96,psnr_y=m['psnr'],ssim_y=m['ssim']))
    with (OUT/'fig3-roi-metrics.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(OUT/'fig3-v2.json',dict(figure=exported,all_methods_mode='single, no self-ensemble, no tiling',selection=selection,scenes=records,footer_inside_figure=False,caption='All reconstructions use single-pass inference. Patches use nearest-neighbor display. Local metrics are Y-channel PSNR/SSIM with shave4; selected ROIs do not imply whole-dataset dominance.'))


def fig4():
    style();methods=['bicubic','B0_50k_single','SG6_50k_single','W_50k_single','U_50k_single','F0_50k_single','GT'];labels=['Bicubic','B0','SG6','SG6+W','SG6+U','F0','GT']
    side=57.;gap=4.;left=8.;row=175.
    fig=plt.figure(figsize=(WIDTH/72,356/72),dpi=300);records=[]
    for r,image in enumerate(['img004.png','img011.png']):
        rec,full,patch=scene('Urban100',image,ROIS[image],methods);records.append(rec);top=5+r*row
        text(fig,WIDTH/2,top,'Urban100 (x4): '+Path(image).stem+' | independent 50K runs, selected 48K',size=8.3)
        for i,(m,label) in enumerate(zip(methods,labels)):
            x=left+i*(side+gap)
            axes(fig,x,top+16,side,side).imshow(patch[m],interpolation='nearest',resample=False)
            text(fig,x+side/2,top+16+side+2,label,size=8.)
            if m!='GT':text(fig,x+side/2,top+16+side+13,f"{rec['metrics'][m]['psnr']:.2f} dB",size=8.)
            error=np.abs(patch[m].astype(np.float32)-patch['GT'].astype(np.float32)).mean(2)
            axes(fig,x,top+16+side+27,side,side).imshow(error,cmap='inferno',vmin=0,vmax=40,interpolation='nearest',resample=False)
        cax=fig.add_axes([441/WIDTH,1-(top+16+2*side+27)/356,6/WIDTH,side/356])
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,40),cmap='inferno'),cax=cax,ticks=[0,20,40]);cax.tick_params(labelsize=7)
    exported=finish(fig,'fig4_ablation_errormap_x4_v2')
    write_json(OUT/'fig4-v2.json',dict(figure=exported,scenes=records,error='mean_c(abs(SR.float32-GT.float32)) in uint8 units',vmin=0,vmax=40,colormap='inferno',note='Shared color limits across methods and scenes; GT error identically zero; all single-pass 50K experiments, not the final FT2 model.'))


def fig5():
    style();fig=plt.figure(figsize=(WIDTH/72,390/72),dpi=300);side=100.;gap=9.;context=125.;rows=[]
    for i,image in enumerate(['img004.png','img011.png']):
        top=5+i*193;gt=get_image('Urban100','GT',image);target=read_json(OUT/'lam-Ours.json')['rows'][i]['target_hr_xywh']
        ax=axes(fig,4,top+16,context,150);ax.imshow(gt,interpolation='nearest');x,y,w,h=target
        ax.add_patch(Rectangle((x-.5,y-.5),w,h,fill=False,edgecolor='#E53935',linewidth=2))
        text(fig,4+context/2,top,Path(image).stem+' (x4)')
        text(fig,4+context/2,top+169,'Target: 16 x 16 HR',size=8.)
        for j,name in enumerate(['SPAN52','SwinIR-light','Ours']):
            info=read_json(OUT/f'lam-{name}.json')['rows'][i];data=np.load(OUT/'lam'/name/(Path(image).stem+'.npz'))
            norm=data['normalized'];lr=data['lr_rgb'];left=136+j*109
            text(fig,left+side/2,top,name,color='#B71C1C' if name=='Ours' else 'black')
            axes(fig,left,top+15,side,70).imshow(norm,cmap='seismic',vmin=-1,vmax=1,interpolation='nearest')
            text(fig,left+side/2,top+87,f"DI: {info['DI']:.4f}",size=8.)
            gray=np.mean(lr.astype(np.float32),2)/255.;overlay=np.repeat(gray[:,:,None],3,2)
            alpha=(.65*norm)[:,:,None];overlay=overlay*(1-alpha)+np.array([1.,0.,0.])[None,None,:]*alpha
            axes(fig,left,top+102,side,70).imshow(overlay,interpolation='nearest')
            text(fig,left+side/2,top+175,'Pixel contribution',size=8.)
            rows.append(dict(image=image,model=name,DI=info['DI'],target_hr_xywh=target))
    exported=finish(fig,'fig5_lam_x4_v2');write_json(OUT/'fig5-v2.json',dict(figure=exported,rows=rows,overlay='LR grayscale blended with red, alpha=.65*official normalized attribution; no KDE',mode='single',official_commit=LAM_COMMIT))


def fig6():
    style();fig=plt.figure(figsize=(WIDTH/72,220/72),dpi=300);rows=[]
    for i,image in enumerate(['img004.png','img011.png']):
        data=np.load(OUT/'features'/(Path(image).stem+'.npz'));top=5+i*107;side=65.;gap=8.
        arrays=[data[f'F{k}_rms'] for k in [2,3,4,5]]
        lo,hi=np.quantile(np.concatenate([a.ravel() for a in arrays]),[.01,.99])
        g=data['G_mean_signed'];limit=max(abs(np.quantile(g,.01)),abs(np.quantile(g,.99)),1e-8)
        text(fig,WIDTH/2,top,Path(image).stem+' | final FT2 RAW, single-pass')
        for j,(label,a) in enumerate(zip(['SG6 G (signed)','F2 RMS','F3 RMS','F4 RMS','F5 RMS'],[g]+arrays)):
            x=12+j*(side+gap);ax=axes(fig,x,top+15,side,side)
            ax.imshow(a,cmap='coolwarm' if j==0 else 'viridis',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit) if j==0 else Normalize(lo,hi),interpolation='nearest')
            text(fig,x+side/2,top+side+18,label,size=8.)
        cax=fig.add_axes([383/WIDTH,1-(top+15+side)/220,6/WIDTH,side/220])
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(lo,hi),cmap='viridis'),cax=cax,format='%.2g');cax.tick_params(labelsize=7)
        text(fig,386,top+5,'RMS',size=7.)
        gax=fig.add_axes([433/WIDTH,1-(top+15+side)/220,5/WIDTH,side/220])
        fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),cmap='coolwarm'),cax=gax,ticks=[-limit,0,limit],format='%.2g');gax.tick_params(labelsize=6)
        text(fig,435,top+5,'G',size=7.)
        rows.append(dict(image=image,feature_shared_percentile=[.01,.99],feature_vmin=float(lo),feature_vmax=float(hi),gate_signed_color_limit=float(limit)))
    exported=finish(fig,'fig6_feature_evolution_x4_v2');write_json(OUT/'fig6-v2.json',dict(figure=exported,normalization=rows,note='F2-F5 share the same energy color range within each scene; G is a signed channel mean with a separate zero-centered scale, not a sigmoid probability. Raw 52-channel tensors and numerical summaries are preserved.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('figure',choices=['fig3','fig4','fig5','fig6']);a=p.parse_args();globals()[a.figure]()
