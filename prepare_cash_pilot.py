"""Export a cash-only ROI experiment with source-grouped train/val/test splits.

Assistant-reviewed boxes remain weak labels. No automatic human confirmation,
no production model changes, no augmentation of validation or held-out images.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import math

ROOT=Path(__file__).resolve().parent

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def checked_box(box,width,height):
    if (not isinstance(box,list) or len(box)!=4 or any(not isinstance(x,(float,int)) or not math.isfinite(x) for x in box)
        or not 0<=box[0]<box[2]<=width or not 0<=box[1]<box[3]<=height):
        raise ValueError(f'Invalid box {box}')
    return box

def validate_rows(review):
    rows=[];groups={};seen=set()
    for row in review['frames']:
        if row.get('review_status')!='assistant_reviewed_for_cash_only' or row.get('complete_for_cash_only') is not True:
            continue
        split={'heldout':'test'}.get(row['split'],row['split'])
        if split not in {'train','val','test'}:raise ValueError('Unknown split')
        key=(row['source_sha256'],row['source_frame'])
        if key in seen:raise ValueError('Duplicate source frame')
        seen.add(key)
        if groups.setdefault(key[0],split)!=split:raise ValueError('Source SHA crosses splits')
        rows.append({**row,'split':split})
    if {r['split'] for r in rows}!={'train','val','test'}:raise ValueError('Three non-empty splits required')
    return rows

def export(review_path,output):
    from PIL import Image
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Choose a new empty dataset directory')
    review=json.loads(Path(review_path).read_text(encoding='utf-8'))
    rows=validate_rows(review)
    output.mkdir(parents=True,exist_ok=True);(output/'images').mkdir()
    sets={s:{'info':{'description':'Cash-only local ROI pilot; assistant-reviewed weak labels; not independent human ground truth'},
             'images':[],'annotations':[],'categories':[{'id':1,'name':'cash'}]} for s in ['train','val','test']}
    sources={};pixels={};ann_id=0
    for iid,row in enumerate(rows,1):
        source=Path(row['source_video_path'])
        if str(source) not in sources:sources[str(source)]=sha(source)
        if sources[str(source)]!=row['source_sha256']:raise ValueError('Original video changed')
        frame_path=Path(row['source_frame_path']).resolve()
        with Image.open(frame_path) as im:
            im=im.convert('RGB');width,height=im.size
            crop=checked_box(row['roi_xyxy'],width,height)
            if any(v!=int(v) for v in crop):raise ValueError('ROI must use integer pixel edges')
            image=im.crop(crop)
        boxes=[checked_box(b,width,height) for b in row['cash_boxes_xyxy']]
        for b in boxes:
            if not crop[0]<=b[0]<b[2]<=crop[2] or not crop[1]<=b[1]<b[3]<=crop[3]:raise ValueError('Cash box extends beyond reviewed ROI')
        digest=hashlib.sha256(image.tobytes()).hexdigest()
        if digest in pixels:raise ValueError(f'Duplicate ROI pixels {pixels[digest]} / {row["source_video"]}')
        pixels[digest]=row['source_video']
        name=f'images/{row["source_video"]}_f{row["source_frame"]:07d}.png'
        image.save(output/name)
        item={'id':iid,'file_name':name,'width':image.width,'height':image.height,
          'source_video':row['source_video'],'source_sha256':row['source_sha256'],'source_group':row['source_sha256'],
          'source_frame':row['source_frame'],'timestamp_sec':row['timestamp_sec'],
          'source_frame_path':str(frame_path),'source_frame_sha256':sha(frame_path),
          'roi_xyxy':crop,'sha256_pixels':digest,'annotation_provenance':'assistant_visual_review_cash_only_roi',
          'human_class_provenance':row.get('human_class_provenance'),'complete_for_cash_only':True,
          'human_geometry_verified':False,'independent_test_eligible':False}
        data=sets[row['split']];data['images'].append(item)
        for b in boxes:
            ann_id+=1;w,h=b[2]-b[0],b[3]-b[1]
            data['annotations'].append({'id':ann_id,'image_id':iid,'category_id':1,'bbox':[b[0]-crop[0],b[1]-crop[1],w,h],
              'area':w*h,'iscrowd':0,'annotation_provenance':'assistant_reviewed_weak_geometry'})
    summary={}
    for split,data in sets.items():
        (output/f'{split}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        positive={a['image_id'] for a in data['annotations']}
        summary[split]={'images':len(data['images']),'cash_boxes':len(data['annotations']),
                       'negative_images':len(data['images'])-len(positive),'source_videos':sorted({i['source_video'] for i in data['images']}),
                       'source_groups':sorted({i['source_sha256'] for i in data['images']})}
    manifest={'version':1,'categories':[{'id':1,'name':'cash'}],'summary':summary,
      'source_review':str(Path(review_path).resolve()),'source_review_sha256':sha(review_path),
      'split_method':'predeclared_whole_video_sha256_groups','test_used_for_training_or_threshold_selection':False,
      'labels':'assistant_reviewed_weak_labels_with_some_prior_user_class_confirmations',
      'model_independent_source_holdout':True,'independent_human_test_ground_truth':False,
      'limitations':['One room; very few examples; frames in one video are correlated.',
        'Only visible cash within each reviewed ROI is exhaustively annotated; other regions/classes are not labels.',
        'A banknote box is not an action or a banking violation.','Food/drink have only one source video; no unseen-food generalization claim.']}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,default=ROOT/'work/cash_pilot_review.json')
    p.add_argument('--output',type=Path,default=ROOT/'training/cash_pilot_20261017/dataset')
    args=p.parse_args();export(args.review,args.output)
