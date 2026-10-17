"""Run the separate experimental cash model in an explicitly selected region.

No reference answers or human annotations are read during inference. Output is
object observations, not automatic verdicts. Production models stay unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from html import escape
from pathlib import Path
from urllib.parse import quote
import run as reporting
from calibrate_threshold import nms
from interaction_refinement import resolve_crop, crop_box_to_source

ROOT=Path(__file__).resolve().parent

def map_cash_predictions(predictions,crop,width,height):
    mapped=[]
    for p in predictions:
        if p['label']!='cash':continue
        x1,y1,x2,y2=p['box']
        if not (0 <= (x1+x2)/2 < crop[2]-crop[0] and 0 <= (y1+y2)/2 < crop[3]-crop[1]):continue
        try:box=crop_box_to_source(p['box'],crop,width,height)
        except ValueError:continue
        mapped.append({**p,'box':box,'crop_box':p['box'],'roi_xyxy':crop,
          'source':'model','model_scope':'experimental_cash_only_roi','violation_established':False})
    return nms(mapped,.6)

def calibrated_threshold(path,weights_sha,override=None):
    if override is not None:value=override
    else:
        if not path.is_file():raise ValueError('Calibration missing; evaluate the model or specify --threshold explicitly')
        calibration=json.loads(path.read_text(encoding='utf-8'))
        if calibration.get('checkpoint_sha256')!=weights_sha:raise ValueError('Calibration belongs to different weights')
        value=calibration['thresholds']['cash']
    if not isinstance(value,(float,int)) or not math.isfinite(value) or not 0<value<1:raise ValueError('threshold must be in (0,1)')
    return value

def make_experiment_index(output):
    links=[]
    for path in sorted(output.glob('*/summary.json')):
        links.append(f'<li><a href="{quote(path.relative_to(output).as_posix())}">{escape(path.parent.name)} — JSON</a></li>')
    (output/'index.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Пробный детектор банкнот</title>'
        '<h1>Пробный детектор банкнот</h1><p>Рамки предметов, не выводы о нарушениях. Превью содержит выбранные кадры. '
        'Время наследует поправки декодера; исходные номера кадров сохранены.</p><ul>'+''.join(links)+'</ul></html>',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    selection=p.add_mutually_exclusive_group(required=True)
    selection.add_argument('--roi',help='Original pixel coordinates x1,y1,x2,y2')
    selection.add_argument('--case-view',action='store_true',help='Use reviewed case band [20,280,400,520], only for 464x848 video')
    p.add_argument('--weights',type=Path,default=ROOT/'training/cash_pilot_20261017/model/best.pth')
    p.add_argument('--calibration',type=Path,default=ROOT/'training/cash_pilot_20261017/evaluation/calibration.json')
    p.add_argument('--threshold',type=float,default=None)
    p.add_argument('--interval',type=float,default=.2)
    p.add_argument('--output',type=Path,default=ROOT/'results/cash_pilot')
    p.add_argument('--device',default=None)
    p.add_argument('--no-video',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if not math.isfinite(args.interval) or args.interval<.04:p.error('interval must be >=.04')
    if not args.weights.is_file():p.error('Experimental weights missing; run training first')
    video=reporting.resolve_video(args.video)
    weights_sha=hashlib.sha256(args.weights.read_bytes()).hexdigest()
    try:args.threshold=calibrated_threshold(args.calibration,weights_sha,args.threshold)
    except (ValueError,KeyError) as error:p.error(str(error))
    provenance={'checkpoint':str(args.weights.resolve()),'checkpoint_sha256':weights_sha,'threshold':args.threshold,
                'code_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                    ('detect_cash_roi.py','train_objects.py','interaction_refinement.py','calibrate_threshold.py')},
                'nms_iou':.6,'preview':'sampled frames; inherited decoder timestamp fallback may apply',
                'roi':args.roi or 'case_view_464x848_[20,280,400,520]','human_reference_matching':False,
                'scope':'experimental_cash_only_roi; not an action or violation model'}
    reporting.PROMPTS['cash_roi_model']=json.dumps(provenance,sort_keys=True)
    args.groups=['cash_roi_model'];args.no_feature_cache=True;args.max_samples=0
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(2)
    from train_objects import load_trained,predict_image
    loaded=load_trained(args.weights,args.device)
    if [c['name'] for c in loaded[2]]!=['cash']:p.error('Expected cash-only checkpoint')
    def detect(image,wrapped,prompt,threshold):
        if args.case_view and image.size!=(464,848):raise ValueError('Case view requires 464x848 source; specify --roi for another view')
        crop=resolve_crop(args.roi or [20,280,400,520],image.width,image.height)
        rows=predict_image(image.crop(crop),wrapped[0],threshold)
        return map_cash_predictions(rows,crop,image.width,image.height)
    summary=reporting.analyze(video,args,(loaded,None,str(loaded[3])),detect)
    summary['cash_model_provenance']=provenance
    summary['limitations'].extend(['This experimental model only observes the explicitly selected ROI.',
        'Training and held-out labels were prepared by an assistant in one room; independent human validation is pending.'])
    reporting.write_json(args.output/video.stem/'summary.json',summary)
    make_experiment_index(args.output)
    print(json.dumps({'status':summary['status'],'output':str(args.output),'samples':summary['analyzed_samples'],
                      'seconds':summary['processing_seconds'],'checkpoint_sha256':weights_sha},ensure_ascii=False))

if __name__=='__main__':main()
