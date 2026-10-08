"""Exercise a frozen NEU model through the live API and keep visual evidence.

Reports box precision/recall at confidence 0.25 and IoU 0.50, not mAP.
Quality-rejected frames are reported separately; no parameters are tuned.
"""
import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']

def request(base, path, content=None, media=None):
    req = urllib.request.Request(base + path, data=content, headers={'Content-Type':media} if media else {})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.status, json.load(response), (time.perf_counter()-started)*1000
    except urllib.error.HTTPError as error:
        return error.code, json.load(error), (time.perf_counter()-started)*1000

def overlap(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0

def run(base, evaluation_path, output, quality_profile='camera'):
    data=ROOT / 'data/processed/neu-det'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'report.json').exists(): raise ValueError('Preserve existing reports; choose a new output directory.')
    evaluation=json.loads(evaluation_path.read_text())
    counter={c:dict(images=0,rejected=0,tp=0,fp=0,fn=0,accepted_gt=0,rejected_gt=0) for c in CLASSES}
    results=[];latencies=[];tiles={};persistence=0
    paths=sorted((data/'images/test').glob('*.jpg'))
    for index,path in enumerate(paths):
        frame=cv2.imread(str(path));height,width=frame.shape[:2]
        truth=[]
        for line in (data/'labels/test'/f'{path.stem}.txt').read_text().splitlines():
            label,x,y,w,h=map(float,line.split())
            truth.append({'label':CLASSES[int(label)],'box':[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]})
        category=path.stem.rsplit('_',1)[0];counter[category]['images']+=1
        raw=path.read_bytes();status,stored,_=request(base,'/api/frames?quality_profile='+quality_profile,raw,'image/jpeg')
        assert status==201,(path,status,stored)
        assert stored['sha256']==hashlib.sha256(raw).hexdigest()
        status,predicted,duration=request(base,f"/api/proxy/frames/{stored['id']}/detect",b'')
        rejected=not stored['quality']['passed']
        if rejected:
            assert status==422,(path,status,predicted)
            counter[category]['rejected']+=1
            for item in truth:counter[item['label']]['rejected_gt']+=1
            detections=[]
        else:
            assert status==200,(path,status,predicted)
            assert predicted['model_sha256']==evaluation['weights_sha256']
            assert predicted['image_sha256']==stored['sha256']
            assert predicted['metrology'] is None and predicted['brake_disc_validated'] is False
            status,persisted,_=request(base,f"/api/proxy/runs/{predicted['id']}")
            assert status==200 and persisted==predicted
            persistence+=1;latencies.append(duration);detections=predicted['detections']
            for item in truth:counter[item['label']]['accepted_gt']+=1
            matched=set()
            for detection in sorted(detections,key=lambda d:d['confidence'],reverse=True):
                label=detection['label'];box=detection['bbox_xyxy_px']
                assert label in CLASSES and len(box)==4 and all(np.isfinite(v) for v in box)
                assert 0<=detection['confidence']<=1
                assert 0<=box[0]<=box[2]<=width+1e-3 and 0<=box[1]<=box[3]<=height+1e-3
                candidates=[(overlap(box,item['box']),i) for i,item in enumerate(truth) if i not in matched and item['label']==label]
                best=max(candidates,default=(0,-1))
                if best[0]>=.5:matched.add(best[1]);counter[label]['tp']+=1
                else:counter[label]['fp']+=1
            for i,item in enumerate(truth):
                if i not in matched:counter[item['label']]['fn']+=1
        results.append({'image':path.name,'quality_passed':not rejected,'ground_truth_boxes':len(truth),
                        'predictions':detections,'frame_id':stored['id'],'run_id':predicted.get('id'),'inference_request_ms':duration})
        if category not in tiles:
            panel=cv2.resize(frame,(300,300));scale=300/width
            for item in truth:
                x1,y1,x2,y2=[round(v*scale) for v in item['box']];cv2.rectangle(panel,(x1,y1),(x2,y2),(0,145,0),1)
            for detection in detections:
                x1,y1,x2,y2=[round(v*scale) for v in detection['bbox_xyxy_px']];cv2.rectangle(panel,(x1,y1),(x2,y2),(0,0,255),2)
            canvas=np.full((355,300,3),245,dtype=np.uint8);canvas[35:335]=panel
            cv2.putText(canvas,category,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(25,25,25),1)
            caption='QUALITY REJECTED' if rejected else f'GT {len(truth)} / predictions {len(detections)}'
            cv2.putText(canvas,caption,(8,350),cv2.FONT_HERSHEY_SIMPLEX,.4,(25,25,25),1);tiles[category]=canvas
        if (index+1)%50==0:print(f'API tested {index+1}/{len(paths)} images',flush=True)
    for values in counter.values():
        values['precision']=values['tp']/(values['tp']+values['fp']) if values['tp']+values['fp'] else None
        values['recall_on_quality_passed_images']=values['tp']/(values['tp']+values['fn']) if values['tp']+values['fn'] else None
    total={k:sum(v[k] for v in counter.values()) for k in ['tp','fp','fn','accepted_gt','rejected_gt','rejected']}
    tests={}
    for name,content,expected in [('invalid_image',b'invalid',422),('oversized_image',b'x'*(8*1024*1024+1),413)]:
        status,_,_=request(base,'/api/frames',content,'image/jpeg');assert status==expected;tests[name]=status
    status,_,_=request(base,'/api/proxy/frames/does-not-exist/detect',b'');assert status==404;tests['missing_frame']=status
    tiles_list=[tiles[c] for c in CLASSES]
    cv2.imwrite(str(output/'examples.jpg'),np.vstack([np.hstack(tiles_list[:3]),np.hstack(tiles_list[3:])]))
    summary={'tested_images':len(paths),'quality_passed':len(paths)-total['rejected'],'quality_rejected':total['rejected'],
        'prediction_runs_verified_in_database':persistence,'model_sha256':evaluation['weights_sha256'],
        'operating_point':{'confidence_threshold':.25,'matching_iou':.5,'matching':'greedy by prediction confidence within class'},
        'box_precision_on_quality_passed_images':total['tp']/(total['tp']+total['fp']) if total['tp']+total['fp'] else None,
        'box_recall_on_quality_passed_images':total['tp']/(total['tp']+total['fn']) if total['tp']+total['fn'] else None,
        'counts':total,'per_class':counter,'negative_request_checks':tests,
        'warm_inference_request_ms':{'median':float(np.median(latencies)),'p95':float(np.quantile(latencies,.95)),
            'scope':'Local HTTP request, image decode, model prediction and DB result commit. Upload/quality request excluded.'},
        'no_model_or_threshold_changes':True,'brake_disc_validated':False,'quality_profile':quality_profile,
        'evaluation_reference':str(evaluation_path.relative_to(ROOT)),
        'fresh_blind_test':False,
        'visual_legend':'Green: ground truth. Red: model predictions. First sorted test image per class, without cherry-picking.'}
    (output/'report.json').write_text(json.dumps({'summary':summary,'images':results},indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base-url',default='http://127.0.0.1:8000')
    parser.add_argument('--evaluation',type=Path,default=ROOT/'models/yolo/neu_yolo11n/evaluation.json')
    parser.add_argument('--output',type=Path,default=ROOT/'data/model_api_test')
    parser.add_argument('--quality-profile',choices=['camera','neu_proxy'],default='camera')
    args=parser.parse_args()
    run(args.base_url,args.evaluation.resolve(),args.output.resolve(),args.quality_profile)
