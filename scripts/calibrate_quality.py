"""Fit dataset proxy profiles on original training only, grouped80/20 calibration holdout."""
import hashlib,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from vision.quality_gate import ROOT,CALIBRATION_DIR,normalized_metrics,rejection_reasons

def training_sources(profile):
    if profile=='neu_proxy':
        base=ROOT/'data/processed/neu-det'; manifest=json.loads((base/'manifest.json').read_text())
        return [{'path':(base/'images/train'/r['image']).relative_to(ROOT).as_posix(),'sha256':r['sha256']} for r in manifest['manifest'] if r['split']=='train']
    manifest=json.loads((ROOT/'data/processed/casting/manifest.json').read_text())
    return [{key:r[key] for key in ('path','sha256')} for r in manifest['records'] if r['split']=='train' and r['label']=='normal']

def split_training(records,seed=42):
    keys=sorted({r['sha256'] for r in records});random.Random(seed).shuffle(keys); fitted=set(keys[:int(len(keys)*.8)])
    return [r for r in records if r['sha256'] in fitted],[r for r in records if r['sha256'] not in fitted]

def read(record):
    content=(ROOT/record['path']).read_bytes()
    if hashlib.sha256(content).hexdigest()!=record['sha256']:raise ValueError('Source image changed since split preparation')
    return cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)

def fit(profile):
    records=training_sources(profile);fitted,validation=split_training(records); measurements=[normalized_metrics(read(r)) for r in fitted]
    def q(key,quantile):return float(np.quantile([r[key] for r in measurements],quantile))
    thresholds={'multiscale_min':q('multiscale_sharpness_ratio',.01)*.9,'blur_ratio_min':q('normalized_laplacian_ratio',.005)*.5,'contrast_min':max(2.,q('contrast_std',.005)*.7),'mean_min':max(5.,q('mean_luminance',.005)-10),'mean_max':min(250.,q('mean_luminance',.995)+5),'saturation_max':min(.9,q('saturation_fraction',.995)+.10),'dark_clipping_max':min(.9,q('dark_clipping_fraction',.995)+.10)}
    def assess(rows):
        counts={key:0 for key in ('clean','gaussian_sigma3','gaussian_sigma5','underexposed','overexposed','blank')}
        for record in rows:
            image=cv2.resize(read(record),(224,224),interpolation=cv2.INTER_AREA)
            versions={'clean':image,'gaussian_sigma3':cv2.GaussianBlur(image,(0,0),3),'gaussian_sigma5':cv2.GaussianBlur(image,(0,0),5),'underexposed':np.clip(image.astype(float)*.12,0,255).astype(np.uint8),'overexposed':np.clip(image.astype(float)+180,0,255).astype(np.uint8),'blank':np.full_like(image,128)}
            for kind,transformed in versions.items():
                rejected=bool(rejection_reasons(normalized_metrics(transformed),thresholds));counts[kind]+=(not rejected) if kind=='clean' else rejected
        return {key:{'count':value,'total':len(rows),'rate':value/len(rows),'meaning':'acceptance' if key=='clean' else 'rejection'} for key,value in counts.items()}
    performance={'fit':assess(fitted),'held_train_validation':assess(validation)}; dimensions=[min(read(record).shape[:2]) for record in fitted]
    metadata={'profile':profile,'version':f'{profile}-quality-v1-seed42','algorithm':'normalized-quality-v1','thresholds':thresholds,'seed':42,'source_split':'original training only; exact-hash grouped80/20 fit/quality-validation','source_dimension_bounds':[min(dimensions),max(dimensions)],'sources':{'fit':fitted,'quality_validation':validation},'source_records_sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),'counts':{'source_training':len(records),'fit':len(fitted),'quality_validation':len(validation)},'selection_rule':'Fixed lower1% multiscale quantile*0.9, lower .5% quantile blur*0.5 contrast*0.7; exposure quantiles +/-10/5 gray units; clipping upper99.5%+0.1. No quality-validation/test threshold optimization.','metrics':{'multiscale_sharpness_ratio':'Laplacian variance ratio of sigma0.7 versus additionally sigma3 smoothed224px grayscale; dimensionless','normalized_laplacian_ratio':'Laplacian variance / max(gray variance,1) on224x224 INTER_AREA grayscale; dimensionless','contrast_std':'224px grayscale standard deviation in0..255 units','mean_luminance':'224px mean grayscale in0..255 units','saturation_fraction':'fraction grayscale>=250; no physical specularity inference','dark_clipping_fraction':'fraction grayscale<=5'},'constraints':{'minimum_clean_holdout_acceptance':.95,'synthetic_corruptions':'Gaussian sigma3/5 at224px, gain0.12 underexposure, +180 clipped overexposure, uniform128 blank'},'performance':performance,'camera_validated':False,'limitations':'Dataset originals treated as clean without operator quality labels. Only severe synthetic degradations evaluated; proxy thresholds do not validate camera quality, mild blur, real glare, or brake discs.'}
    if performance['held_train_validation']['clean']['rate']<.95:raise RuntimeError(f'{profile} failed clean held-subset acceptance constraint; calibration not deployed')
    CALIBRATION_DIR.mkdir(parents=True,exist_ok=True);(CALIBRATION_DIR/f'{profile}.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    print(json.dumps({'profile':profile,'counts':metadata['counts'],'thresholds':thresholds,'performance':performance},indent=2),flush=True)
    return metadata

if __name__=='__main__':
    cv2.setNumThreads(4)
    for profile in ('neu_proxy','casting_proxy'):fit(profile)
