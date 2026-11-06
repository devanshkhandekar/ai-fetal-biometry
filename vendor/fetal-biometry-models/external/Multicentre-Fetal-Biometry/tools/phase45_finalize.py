#!/usr/bin/env python
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path.cwd()
DEFAULTS = {
    'TAD': {'csv': ROOT/'data/annotations/FP/Abdomen_Test.csv','images': ROOT/'data/images/FP/Abdomen','pred': ROOT/'output/FETAL/local_FP_abdomen_TAD/predictions.pth','prefix':'tad'},
    'APAD': {'csv': ROOT/'data/annotations/FP/Abdomen_Test.csv','images': ROOT/'data/images/FP/Abdomen','pred': ROOT/'output/FETAL/local_FP_abdomen_APAD/predictions.pth','prefix':'apad'},
    'FL': {'csv': ROOT/'data/annotations/FP/Femur_Test.csv','images': ROOT/'data/images/FP/Femur','pred': ROOT/'output/FETAL/local_FP_femur_FL/predictions.pth','prefix':'fl'},
}

def drop_index_columns(df):
    junk=[c for c in df.columns if str(c).lower()=='index' or str(c).startswith('Unnamed')]
    return df.drop(columns=junk) if junk else df

def load_predictions(path):
    if not path.exists():
        raise FileNotFoundError(f'Missing predictions: {path}')
    obj=torch.load(path,map_location='cpu')
    arr=obj.detach().cpu().numpy() if isinstance(obj,torch.Tensor) else np.asarray(obj)
    if arr.ndim!=3 or arr.shape[1:]!=(2,2):
        raise ValueError(f'Expected (N,2,2), got {arr.shape}')
    return arr.astype(np.float64)

def dataset_aligned_rows(csv_path,prefix):
    df=drop_index_columns(pd.read_csv(csv_path)).copy()
    image_col=df.columns[0]
    coords=[f'{prefix}_1_x',f'{prefix}_1_y',f'{prefix}_2_x',f'{prefix}_2_y']
    xyz=df[coords].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    bad_image=df[image_col].isna().to_numpy()
    bad_landmark=((xyz<0)|~np.isfinite(xyz)).any(axis=1)
    out=df.loc[~(bad_image|bad_landmark)].reset_index(drop=True)
    return out.rename(columns={image_col:'image_name'})

def px_column(df):
    if 'px_to_mm_rate' in df.columns: return 'px_to_mm_rate'
    if 'px_to_mm' in df.columns: return 'px_to_mm'
    raise KeyError('No px_to_mm_rate/px_to_mm column')

def distance(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

def measurement_table(metric):
    cfg=DEFAULTS[metric]; prefix=cfg['prefix']
    df=dataset_aligned_rows(cfg['csv'],prefix)
    preds=load_predictions(cfg['pred'])
    if len(df)!=len(preds):
        raise RuntimeError(f'{metric}: filtered CSV rows={len(df)}, predictions={len(preds)}')
    px=pd.to_numeric(df[px_column(df)],errors='coerce').to_numpy(float)
    gt1x=pd.to_numeric(df[f'{prefix}_1_x'],errors='coerce').to_numpy(float)
    gt1y=pd.to_numeric(df[f'{prefix}_1_y'],errors='coerce').to_numpy(float)
    gt2x=pd.to_numeric(df[f'{prefix}_2_x'],errors='coerce').to_numpy(float)
    gt2y=pd.to_numeric(df[f'{prefix}_2_y'],errors='coerce').to_numpy(float)
    p1x,p1y,p2x,p2y=preds[:,0,0],preds[:,0,1],preds[:,1,0],preds[:,1,1]
    gt_px=distance(gt1x,gt1y,gt2x,gt2y); pr_px=distance(p1x,p1y,p2x,p2y)
    valid=np.isfinite(px)&(px>0)
    out=pd.DataFrame({'image_name':df['image_name'].astype(str),'px_to_mm_rate':px,
        'gt_p1_x':gt1x,'gt_p1_y':gt1y,'gt_p2_x':gt2x,'gt_p2_y':gt2y,
        'pred_p1_x':p1x,'pred_p1_y':p1y,'pred_p2_x':p2x,'pred_p2_y':p2y,
        'gt_distance_px':gt_px,'pred_distance_px':pr_px,'has_valid_mm':valid})
    out['gt_mm']=np.where(valid,gt_px*px,np.nan); out['pred_mm']=np.where(valid,pr_px*px,np.nan)
    out['signed_error_mm']=out['pred_mm']-out['gt_mm']; out['abs_error_mm']=np.abs(out['signed_error_mm'])
    return out

def summarize(df):
    d=df[np.isfinite(df.abs_error_mm)&np.isfinite(df.signed_error_mm)]
    ae=d.abs_error_mm.to_numpy(float); se=d.signed_error_mm.to_numpy(float)
    if len(d)==0: raise RuntimeError('No calibrated rows')
    return {'n':int(len(d)),'mae_mm':float(ae.mean()),'median_abs_error_mm':float(np.median(ae)),
      'rmse_mm':float(np.sqrt(np.mean(se**2))),'bias_mm':float(se.mean()),
      'signed_error_std_mm':float(se.std(ddof=1)) if len(se)>1 else 0.0,
      'p90_abs_error_mm':float(np.percentile(ae,90)),'p95_abs_error_mm':float(np.percentile(ae,95)),
      'max_abs_error_mm':float(ae.max())}

def save_json(x,p):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2))

def overlay(df,image_dir,out_dir,label,top_n=20):
    out_dir.mkdir(parents=True,exist_ok=True)
    d=df[np.isfinite(df.abs_error_mm)]
    if d.empty:return
    sel=pd.concat([d.nlargest(min(top_n,len(d)),'abs_error_mm'),d.nsmallest(min(5,len(d)),'abs_error_mm')]).drop_duplicates('image_name')
    for _,r in sel.iterrows():
        ip=image_dir/str(r.image_name)
        if not ip.exists(): continue
        im=Image.open(ip).convert('RGB'); fig,ax=plt.subplots(figsize=(8,6)); ax.imshow(im)
        ax.plot([r.gt_p1_x,r.gt_p2_x],[r.gt_p1_y,r.gt_p2_y],linewidth=2,label='GT')
        ax.scatter([r.gt_p1_x,r.gt_p2_x],[r.gt_p1_y,r.gt_p2_y],s=30)
        ax.plot([r.pred_p1_x,r.pred_p2_x],[r.pred_p1_y,r.pred_p2_y],linewidth=2,label='Pred')
        ax.scatter([r.pred_p1_x,r.pred_p2_x],[r.pred_p1_y,r.pred_p2_y],s=30)
        ax.set_title(f'{label} | GT={r.gt_mm:.2f} mm | Pred={r.pred_mm:.2f} mm | AE={r.abs_error_mm:.2f} mm')
        ax.axis('off'); ax.legend(loc='lower right'); fig.tight_layout(); fig.savefig(out_dir/f'{Path(str(r.image_name)).stem}_{label}.png',dpi=140,bbox_inches='tight'); plt.close(fig)

def plots(df,out_dir,label):
    out_dir.mkdir(parents=True,exist_ok=True); d=df[np.isfinite(df.abs_error_mm)]
    if d.empty:return
    fig,ax=plt.subplots(figsize=(7,5)); ax.hist(d.abs_error_mm.to_numpy(float),bins=25); ax.set_xlabel('Absolute error (mm)'); ax.set_ylabel('Count'); ax.set_title(f'{label}: absolute error'); fig.tight_layout(); fig.savefig(out_dir/f'{label}_error_histogram.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,6)); ax.scatter(d.gt_mm,d.pred_mm,alpha=.6); lo=min(d.gt_mm.min(),d.pred_mm.min()); hi=max(d.gt_mm.max(),d.pred_mm.max()); ax.plot([lo,hi],[lo,hi]); ax.set_xlabel('GT (mm)'); ax.set_ylabel('Prediction (mm)'); ax.set_title(f'{label}: GT vs prediction'); fig.tight_layout(); fig.savefig(out_dir/f'{label}_gt_vs_pred.png',dpi=160); plt.close(fig)
    mean=(d.gt_mm+d.pred_mm)/2; diff=d.pred_mm-d.gt_mm; bias=diff.mean(); sd=diff.std(ddof=1)
    fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(mean,diff,alpha=.6); ax.axhline(bias); ax.axhline(bias+1.96*sd,linestyle='--'); ax.axhline(bias-1.96*sd,linestyle='--'); ax.set_xlabel('Mean measurement (mm)'); ax.set_ylabel('Pred-GT (mm)'); ax.set_title(f'{label}: Bland-Altman'); fig.tight_layout(); fig.savefig(out_dir/f'{label}_bland_altman.png',dpi=160); plt.close(fig)

def run_abdomen():
    od=ROOT/'phase45_validation/phase4_abdomen'; od.mkdir(parents=True,exist_ok=True)
    tad=measurement_table('TAD'); apad=measurement_table('APAD')
    tad.to_csv(od/'TAD_predictions.csv',index=False); apad.to_csv(od/'APAD_predictions.csv',index=False)
    tm,am=summarize(tad),summarize(apad); save_json(tm,od/'TAD_metrics.json'); save_json(am,od/'APAD_metrics.json')
    overlay(tad,DEFAULTS['TAD']['images'],od/'overlays/TAD','TAD'); overlay(apad,DEFAULTS['APAD']['images'],od/'overlays/APAD','APAD')
    plots(tad,od/'plots','TAD'); plots(apad,od/'plots','APAD')
    t=tad[['image_name','gt_mm','pred_mm','has_valid_mm']].rename(columns={'gt_mm':'gt_tad_mm','pred_mm':'pred_tad_mm','has_valid_mm':'tad_valid_mm'})
    a=apad[['image_name','gt_mm','pred_mm','has_valid_mm']].rename(columns={'gt_mm':'gt_apad_mm','pred_mm':'pred_apad_mm','has_valid_mm':'apad_valid_mm'})
    ac=t.merge(a,on='image_name',how='inner'); factor=math.pi/2
    ac['has_valid_mm']=ac.tad_valid_mm&ac.apad_valid_mm; ac['gt_ac_mm']=factor*(ac.gt_tad_mm+ac.gt_apad_mm); ac['pred_ac_mm']=factor*(ac.pred_tad_mm+ac.pred_apad_mm); ac['signed_error_mm']=ac.pred_ac_mm-ac.gt_ac_mm; ac['abs_error_mm']=np.abs(ac.signed_error_mm)
    ac.to_csv(od/'abdomen_AC_predictions.csv',index=False)
    ace=ac.rename(columns={'gt_ac_mm':'gt_mm','pred_ac_mm':'pred_mm'}); acm=summarize(ace); acm['formula']='AC = pi/2 * (TAD + APAD)'; acm['note']='GT AC is derived from annotated TAD/APAD, not an independently annotated AC contour.'; save_json(acm,od/'AC_metrics.json'); plots(ace,od/'plots','AC')
    print('\nPHASE 4 ABDOMEN'); print('TAD',json.dumps(tm,indent=2)); print('APAD',json.dumps(am,indent=2)); print('AC',json.dumps(acm,indent=2)); print('Saved to',od)

def run_femur():
    od=ROOT/'phase45_validation/phase5_femur'; od.mkdir(parents=True,exist_ok=True)
    fl=measurement_table('FL'); fl.to_csv(od/'FL_predictions.csv',index=False); m=summarize(fl); save_json(m,od/'FL_metrics.json'); overlay(fl,DEFAULTS['FL']['images'],od/'overlays/FL','FL'); plots(fl,od/'plots','FL'); print('\nPHASE 5 FEMUR'); print(json.dumps(m,indent=2)); print('Saved to',od)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('task',choices=['abdomen','femur']); a=ap.parse_args(); run_abdomen() if a.task=='abdomen' else run_femur()
if __name__=='__main__': main()
