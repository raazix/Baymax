"""Explicit camera/proxy capture quality with content-addressed calibration."""
from functools import lru_cache
import hashlib,json,math
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CALIBRATION_DIR=ROOT/'models/quality'
class QualityCalibrationUnavailable(RuntimeError): pass
# Generic, resolution-normalised limits for live/unknown cameras. Sharpness limits are the more permissive of the two dataset calibrations.
CAMERA_THRESHOLDS={'multiscale_min':16.0,'blur_ratio_min':.007,'contrast_min':12.0,'mean_min':25.0,'mean_max':235.0,'saturation_max':.85,'dark_clipping_max':.6}

def validate_frame(frame):
    if not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3 or min(frame.shape[:2])<8:
        raise ValueError('Expected BGR uint8 image with three channels, dimensions at least8 pixels')

def normalized_metrics(frame):
    validate_frame(frame)
    gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(224,224),interpolation=cv2.INTER_AREA).astype(np.float64)
    smooth=cv2.GaussianBlur(gray,(0,0),.7); coarse=cv2.GaussianBlur(smooth,(0,0),3)
    multiscale=float(cv2.Laplacian(smooth,cv2.CV_64F).var()/max(float(cv2.Laplacian(coarse,cv2.CV_64F).var()),.0001))
    return {'multiscale_sharpness_ratio':multiscale,'normalized_laplacian_ratio':float(cv2.Laplacian(gray,cv2.CV_64F).var()/max(float(gray.var()),1.)), 'contrast_std':float(gray.std()),'mean_luminance':float(gray.mean()),'saturation_fraction':float(np.mean(gray>=250)), 'dark_clipping_fraction':float(np.mean(gray<=5))}

@lru_cache(maxsize=8)
def _parse(digest,content):
    result=json.loads(content)
    if not isinstance(result,dict) or result.get('algorithm')!='normalized-quality-v1': raise QualityCalibrationUnavailable('Unsupported calibration algorithm')
    try:
        thresholds=result['thresholds']
        keys=('multiscale_min','blur_ratio_min','contrast_min','mean_min','mean_max','saturation_max','dark_clipping_max')
        if not isinstance(thresholds,dict) or any(isinstance(thresholds[key],bool) or not isinstance(thresholds[key],(int,float)) or not math.isfinite(thresholds[key]) for key in keys):
            raise ValueError('Expected finite numeric thresholds')
        if any(thresholds[key]<0 for key in ('multiscale_min','blur_ratio_min','contrast_min')) or not 0<=thresholds['mean_min']<thresholds['mean_max']<=255 or any(not 0<=thresholds[key]<=1 for key in ('saturation_max','dark_clipping_max')):
            raise ValueError('Thresholds outside supported ranges')
        bounds=result['source_dimension_bounds']
        if not isinstance(bounds,list) or len(bounds)!=2 or any(isinstance(value,bool) or not isinstance(value,int) or value<8 for value in bounds) or bounds[0]>bounds[1]:
            raise ValueError('Invalid dimensions')
        if not isinstance(result['version'],str) or not result['version'] or not isinstance(result['limitations'],str):
            raise ValueError('Missing calibration identity or limitations')
    except (KeyError,TypeError,ValueError) as exc:
        raise QualityCalibrationUnavailable('Invalid quality calibration metadata') from exc
    return result

def load_profile(profile):
    try:
        content=(CALIBRATION_DIR/f'{profile}.json').read_bytes(); digest=hashlib.sha256(content).hexdigest(); metadata=_parse(digest,content)
    except (OSError,ValueError,KeyError) as exc: raise QualityCalibrationUnavailable(f'Quality profile unavailable: {profile}; run scripts/calibrate_quality.py') from exc
    if metadata.get('profile')!=profile: raise QualityCalibrationUnavailable('Calibration profile identity mismatch')
    return metadata,digest

def profile_status():
    profiles={'camera':{'configured':True,'calibrated':False,'calibration_version':'camera-generic-v2','calibration_sha256':None}}
    for name in ('neu_proxy','casting_proxy'):
        try:
            metadata,digest=load_profile(name)
            profiles[name]={'configured':True,'calibrated':True,'calibration_version':metadata['version'],'calibration_sha256':digest,'scope':'dataset training only; camera unvalidated'}
        except QualityCalibrationUnavailable:
            profiles[name]={'configured':False,'calibrated':False,'scope':'dataset proxy'}
    return profiles

def rejection_reasons(metrics,thresholds):
    reasons=[]
    for field,direction,key,reason in [('multiscale_sharpness_ratio','min','multiscale_min','severe_blur_multiscale'),('normalized_laplacian_ratio','min','blur_ratio_min','severe_blur_or_insufficient_texture'),('contrast_std','min','contrast_min','insufficient_contrast_or_blank'),('mean_luminance','min','mean_min','underexposure'),('mean_luminance','max','mean_max','overexposure'),('saturation_fraction','max','saturation_max','excessive_bright_clipping'),('dark_clipping_fraction','max','dark_clipping_max','excessive_dark_clipping')]:
        if (direction=='min' and metrics[field]<thresholds[key]) or (direction=='max' and metrics[field]>thresholds[key]): reasons.append(reason)
    return reasons

def check_frame(frame,profile='camera'):
    validate_frame(frame)
    if profile not in ('camera','neu_proxy','casting_proxy'): raise ValueError(f'Unknown quality profile: {profile}')
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); blur=float(cv2.Laplacian(gray,cv2.CV_64F).var()); saturation=float(np.mean(gray>=250))
    common={'profile':profile,'laplacian_variance':round(blur,2),'specular_fraction':round(saturation,4),'saturation_fraction':round(saturation,4),'saturation_note':'Grayscale fraction >=250; not physical specular reflection. specular_fraction is a compatibility alias.'}
    if profile=='camera':
        metrics=normalized_metrics(frame); reasons=rejection_reasons(metrics,CAMERA_THRESHOLDS)
        return {**common,'passed':not reasons,'thresholds':CAMERA_THRESHOLDS,'normalized_metrics':metrics,'rejection_reasons':reasons,'calibration_sha256':None,'calibration_version':'camera-generic-v2','calibrated':False,'note':'Generic resolution-normalised limits: rejects blank, black, blown-out and genuinely blurred frames only. Not calibrated to any camera or lighting; exposure windows are intentionally wide.'}
    metadata,digest=load_profile(profile); metrics=normalized_metrics(frame); reasons=rejection_reasons(metrics,metadata['thresholds']); low,high=metadata['source_dimension_bounds']
    if not low<=min(frame.shape[:2])<=high or not .8<=frame.shape[1]/frame.shape[0]<=1.25: reasons.append('dimensions_outside_calibrated_proxy_domain')
    return {**common,'passed':not reasons,'thresholds':metadata['thresholds'],'normalized_metrics':metrics,'rejection_reasons':reasons,'calibration_sha256':digest,'calibration_version':metadata['version'],'calibrated':True,'calibration_scope':'train-only dataset proxy; not camera/brake-disc validation','note':metadata['limitations']}


# Lighting/size limits calibrated on one dataset are not defensible for arbitrary uploads. For images that fail only
# those dataset-specific checks, apply generic sanity limits instead. Sharpness and texture checks still come from
# the calibrated profile and are never waived.
GENERIC_LIMITS = {
    'dimensions_outside_calibrated_proxy_domain': lambda m: True,
    'underexposure': lambda m: m['mean_luminance'] >= 25,
    'overexposure': lambda m: m['mean_luminance'] <= 235,
    'excessive_bright_clipping': lambda m: m['saturation_fraction'] <= .85,
    'excessive_dark_clipping': lambda m: m['dark_clipping_fraction'] <= .6,
    'insufficient_contrast_or_blank': lambda m: m['contrast_std'] >= 12,
}

def check_frame_open_domain(frame, profile='camera'):
    result = check_frame(frame, profile)
    if result['passed'] or profile == 'camera' or 'normalized_metrics' not in result:
        return result
    metrics = result['normalized_metrics']
    remaining = [reason for reason in result['rejection_reasons'] if not GENERIC_LIMITS.get(reason, lambda m: False)(metrics)]
    if remaining:
        return {**result, 'rejection_reasons': remaining}
    return {**result, 'passed': True, 'waived_checks': result['rejection_reasons'], 'rejection_reasons': [],
            'profile_note': 'Image differs from the dataset the quality gate was calibrated on (size and/or lighting), so those '
                            'dataset-specific limits were waived and generic limits applied. Sharpness and blank-image checks still passed. '
                            'Model results on non-dataset images are not validated.'}
