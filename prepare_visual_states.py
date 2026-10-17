"""Build the reviewed local state-reference model and video-disjoint evaluation.

Annotations below are manual visual readings of saved full frames and enlarged
safe/door crops. They are not automatically inferred from names or a prior model.
Only these sparse reviewed timestamps have ground truth; intervening states are
not assumed. Split/threshold selection is recorded in the output manifest.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np
from visual_states import ANCHORS, ROIS, ROOT, DEFAULT_MODEL, BASELINE_MODEL, MULTIVIEW_MODEL, SceneRegistration, VisualStateDetector, atomic_json, patch_feature, sha256

TRAIN = (20, 21, 22, 24, 25, 26, 28, 30, 33)
VALIDATION = (23, 27, 29, 31, 32)
# Labels correspond to the 0, 2, 4, ... second contact-sheet frames, in order.
# c = visibly closed; o = visibly open; u = indeterminate/obscured.
ANNOTATIONS = {
    20: ('cccccc',          'cooooo'),
    21: ('cccccccccc',      'cocccccccc'),
    22: ('ccccccccouc',     'ccccccccccc'),
    23: ('cccccccou',       'ccccccccc'),
    24: ('ccccccc',         'ccccccc'),
    25: ('cccccccccccccooc','cccccooooocccccc'),
    26: ('cccccccouc',      'cccccccccc'),
    27: ('ccccccccccuoc',   'ccccccocccccc'),
    28: ('ccccccoc',        'cccccccc'),
    29: ('ccccccouc',       'ccccccccc'),
    30: ('ccooooooo',       'uuuuuuuuu'),
    31: ('cccooccccccc',    'uuuuuuuuuuuu'),
    32: ('ccooooooooo',     'uuuuuuuuuuu'),
    33: ('ccooooo',         'uuuuuuu'),
}
# X24 at 10 s is visibly open. Kept as an explicit review correction rather
# than silently deriving a label from neighbouring frames.
ANNOTATIONS[24] = ('cccccoc', 'ccccccc')
LABEL = {'c': 'closed', 'o': 'open', 'u': 'unknown'}


def number(path):
    return int(re.search(r'\d+', path.stem).group())


def choose_thresholds(features, labels, videos):
    """Whole-training-video leave-one-out, never the held-out evaluation split."""
    predictions = []
    for i in range(len(features)):
        available = np.asarray(videos) != videos[i]
        distances = np.sqrt(np.mean((features[available] - features[i]) ** 2, axis=1))
        candidate_labels = labels[available]
        j = int(distances.argmin())
        state, distance = str(candidate_labels[j]), float(distances[j])
        alternatives = distances[candidate_labels != state]
        competing = float(alternatives.min()) if len(alternatives) else 1.
        predictions.append(dict(truth=str(labels[i]), nearest=state, distance=distance,
                                margin=(competing-distance)/max(competing, 1e-9), video=int(videos[i])))
    options = []
    for maximum in [.005, .01, .02, .035, .05, .065, .08, .10, .12, .15]:
        for margin in [.1, .2, .3, .4, .5]:
            accepted = [row for row in predictions if row['nearest'] != 'unknown' and row['distance'] <= maximum and row['margin'] >= margin]
            errors = sum(row['nearest'] != row['truth'] for row in accepted)
            coverage = []
            for state in ('closed', 'open'):
                total = sum(row['truth'] == state for row in predictions)
                coverage.append(sum(row['truth'] == state and row['nearest'] == state for row in accepted) / max(total, 1))
            if errors == 0:
                options.append((sum(coverage), len(accepted), -maximum, margin, maximum))
    # If no error-free threshold exists, abstain rather than silently using an
    # unvalidated acceptance radius. This does not rebuild the preserved v1.
    best = max(options) if options else (0, 0, -1e-6, 1., 1e-6)
    return dict(max_distance=best[4], min_margin=best[3]), dict(method='whole_training_video_leave_one_out',
                examples=len(predictions), accepted=best[1], accepted_errors=0, macro_state_coverage=best[0]/2,
                observations=predictions)


def build(output: Path = BASELINE_MODEL):
    cv2.setNumThreads(1)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    frames_root = ROOT / 'work' / 'visual_states' / 'frames'
    source_reference = frames_root / 'X24_000.0.jpg'
    if not source_reference.exists():
        raise ValueError('Reviewed reference frame is missing. Extract/review X24 at 0 s first.')
    shutil.copy2(source_reference, output / 'reference.jpg')
    registration = SceneRegistration(cv2.imread(str(source_reference)))
    videos = {number(p): p for p in (ROOT / 'videos').glob('*.mov')}
    arrays = {kind: [] for kind in ROIS}; labels = {kind: [] for kind in ROIS}; groups = {kind: [] for kind in ROIS}
    annotations, skipped = [], []
    for n in (*TRAIN, *VALIDATION):
        path = videos[n]; digest = sha256(path)
        door_labels, safe_labels = ANNOTATIONS[n]
        if len(door_labels) != len(safe_labels):
            raise ValueError(f'Unequal label counts for X{n}')
        for idx, (door, safe) in enumerate(zip(door_labels, safe_labels)):
            second = idx * 2
            file = frames_root / f'X{n}_{second:05.1f}.jpg'
            frame = cv2.imread(str(file))
            if frame is None:
                raise ValueError(f'Reviewed frame missing: {file}')
            frame_file = output / 'reviewed_frames' / file.name
            frame_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, frame_file)
            row = dict(video=path.name, video_number=n, video_sha256=digest, frame_index=second*30,
                       timestamp_sec=second, image=str(frame_file.relative_to(output)), image_sha256=sha256(frame_file),
                       split='train' if n in TRAIN else 'validation', door=LABEL[door], safe=LABEL[safe],
                       annotation_source='assistant_visual_review_full_frame_and_state_crop')
            annotations.append(row)
            if n not in TRAIN:
                continue
            aligned, transform, info = registration.align(frame)
            row['registration'] = info
            if aligned is None:
                skipped.append(dict(video=n, timestamp=second, reason=info['reason']))
                continue
            for kind, box in ROIS.items():
                x1,y1,x2,y2=box
                if np.mean(transform[1][y1:y2,x1:x2] == 255) < .98:
                    continue
                arrays[kind].append(patch_feature(aligned[y1:y2,x1:x2])); labels[kind].append(row[kind]); groups[kind].append(n)
    thresholds, cross_validation, data = {}, {}, {}
    for kind in ROIS:
        feats, classes = np.asarray(arrays[kind], dtype=np.float32), np.asarray(labels[kind], dtype='<U7')
        thresholds[kind], cross_validation[kind] = choose_thresholds(feats, classes, groups[kind])
        data[kind+'_features']=feats; data[kind+'_labels']=classes; data[kind+'_source_videos']=np.asarray(groups[kind],dtype=np.int16)
    np.savez_compressed(output/'examples.npz', **data)
    atomic_json(output/'annotations.json', dict(schema_version=1, train_videos=list(TRAIN), validation_videos=list(VALIDATION),
        split_method='whole_video', frames=annotations, label_legend=LABEL,
        limitations=['Sparse frames reviewed every two seconds; this is not dense event ground truth.',
                     'Single room, limited people and camera variations; no independent new-room evaluation.',
                     'Unknown safe examples with the counter in front are deliberately conservative.']))
    atomic_json(output/'model.json', dict(schema_version=1, reference_file='reference.jpg', reference_sha256=sha256(output/'reference.jpg'),
        examples_file='examples.npz', examples_sha256=sha256(output/'examples.npz'),
        thresholds=thresholds, rois=ROIS, anchor_regions=ANCHORS, train_videos=list(TRAIN), validation_videos=list(VALIDATION),
        annotation_sha256=sha256(output/'annotations.json'), inference_uses_video_identity=False,
        classifier='nearest reviewed appearance with distance/margin rejection', locked_state_supported=False))
    atomic_json(output/'training_cross_validation.json',cross_validation)
    print(json.dumps(dict(thresholds=thresholds,counts={k:dict(Counter(labels[k])) for k in ROIS},skipped=skipped),ensure_ascii=False,indent=2))
    return annotations


def evaluate(output: Path = BASELINE_MODEL):
    output = Path(output)
    detector = VisualStateDetector(output)
    annotations = json.loads((output/'annotations.json').read_text(encoding='utf-8'))
    train = set(annotations['train_videos']); validation = set(annotations['validation_videos'])
    if train & validation:
        raise ValueError('Train/validation video leakage.')
    predictions=[]
    for row in annotations['frames']:
        if row['split'] != 'validation': continue
        frame=cv2.imread(str(output/row['image']))
        result=detector.predict(frame)
        predictions.append(dict(**row,prediction=result))
    metrics={}
    for kind in ROIS:
        confusion={truth:{state:0 for state in ('closed','open','unknown')} for truth in ('closed','open','unknown')}
        for row in predictions:confusion[row[kind]][row['prediction'][kind]['state']]+=1
        totals={truth:sum(counts.values()) for truth,counts in confusion.items()}
        supported=sum(totals[s] for s in ('closed','open'))
        known=sum(confusion[truth][state] for truth in confusion for state in ('closed','open'))
        correct=sum(confusion[s][s] for s in ('closed','open'))
        tp=confusion['open']['open']; fp=confusion['closed']['open']+confusion['unknown']['open']; fn=totals['open']-tp
        metrics[kind]=dict(confusion=confusion,annotated_frames=len(predictions),known_state_predictions=known,
            accuracy_when_known=correct/known if known else None,coverage_of_known_truth=correct/supported if supported else None,
            open_tp=tp,open_fp=fp,open_fn_including_unknown=fn,open_precision=tp/(tp+fp) if tp+fp else None,
            open_recall=tp/(tp+fn) if tp+fn else None)
    # With the static anchors intact, completely occluding either state ROI must
    # not produce a confident state. These are synthetic diagnostics, not test
    # examples used to choose thresholds.
    reference=cv2.imread(str(output/'reference.jpg'))
    occlusion=[]
    for kind,box in ROIS.items():
        for color in ((0,0,0),(240,240,240),(150,90,180)):
            frame=reference.copy();x1,y1,x2,y2=box;frame[y1:y2,x1:x2]=color
            result=detector.predict(frame)
            occlusion.append(dict(object=kind,color=color,state=result[kind]['state'],reason=result[kind]['reason']))
    novelty=detector.predict(np.zeros_like(reference))
    report=dict(schema_version=1,protocol='Held-out whole videos; thresholds selected only by leave-one-training-video-out.',
        train_videos=list(TRAIN),validation_videos=list(VALIDATION),dataset_sha256=sha256(output/'annotations.json'),
        metrics=metrics, synthetic_full_roi_occlusion=occlusion,blank_scene_result={k:novelty[k]['state'] for k in ROIS},
        predictions=predictions,limitations=annotations['limitations'],
        event_metrics='Not estimated from sparse frame labels. Event intervals need separate dense video review.')
    atomic_json(output/'validation_report.json',report)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    print('synthetic_occlusion',json.dumps(occlusion))
    return report


# Additional camera poses reviewed independently of the five legacy evaluation
# videos. New evaluation videos X2/X37/X46 never supply model features.
MULTIVIEW_SPECS = [
    dict(view_id='desk_back_left', reference='X1_000.0.jpg', videos=[1, 3],
         door_labels={1: 'coocccccccccc', 3: 'cooccccccc'},
         rois={'door': [50, 50, 105, 210]},
         anchors=[[100, 40, 210, 205], [245, 50, 460, 355], [370, 600, 460, 845]],
         reflection=[0, 210, 62, 305]),
    dict(view_id='desk_back_right', reference='X34_000.0.jpg', videos=[34, 38],
         door_labels={34: 'ccoccccoc', 38: ''.join('o' if t in (2, 4, 26, 34, 54) else 'c' for t in range(0, 58, 2))},
         rois={'door': [90, 7, 165, 175]},
         anchors=[[175, 5, 280, 165], [300, 20, 460, 330], [360, 650, 460, 840]],
         reflection=[24, 175, 96, 260]),
    dict(view_id='desk_front_white', reference='X43_000.0.jpg', videos=[43, 44, 45],
         door_labels={43: 'c' * 12, 44: 'c' * 10, 45: 'c' * 17},
         rois={'door': [80, 28, 145, 193]},
         anchors=[[160, 20, 255, 220], [280, 70, 460, 330], [370, 620, 460, 840]],
         reflection=[48, 210, 120, 278]),
]
DENSE_DOOR_LABELS = {
    1: {1.5:'o', 2.5:'o', 3.5:'o', 4.5:'c', 5.5:'c'},
    3: {1.5:'o', 2.5:'o', 3.5:'o', 4.5:'c', 5.5:'c'},
    34: {2.5:'o', 3.5:'o', 4.5:'o', 5.5:'c', 6.5:'c', 12.5:'c', 13.5:'o', 14.5:'o', 15.5:'c', 16.5:'c'},
    38: {.5:'c',1.5:'c',2.5:'o',3.5:'o',4.5:'c',5.5:'c',24.5:'c',25.5:'c',26.5:'o',27.5:'o',
         32.5:'o',33.5:'o',34.5:'o',35.5:'c',36.5:'c',37.5:'c',50.5:'c',51.5:'c',53.5:'o',54.5:'o',55.5:'o',56.5:'c'},
}


def build_multiview(output: Path = MULTIVIEW_MODEL):
    """Build CPU appearance references; leave the earlier model byte-for-byte intact."""
    cv2.setNumThreads(1)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    frames_root = ROOT / 'work' / 'states_v2' / 'frames'
    views, annotations, cv_reports = [], [], {}
    legacy = output / 'views' / 'legacy'; legacy.mkdir(parents=True, exist_ok=True)
    for name in ['reference.jpg', 'examples.npz']:
        shutil.copy2(BASELINE_MODEL / name, legacy / name)
    legacy_meta = json.loads((BASELINE_MODEL / 'model.json').read_text(encoding='utf-8'))
    legacy_meta['reflection_regions'] = [dict(region_id='glass_behind_cashier',
        polygon=[[68, 220], [119, 220], [119, 277], [68, 277]])]
    atomic_json(legacy / 'model.json', legacy_meta)
    views.append(dict(view_id='legacy', model_file='views/legacy/model.json', model_sha256=sha256(legacy/'model.json')))
    # Compatibility image for human inspection and the original occlusion tests.
    shutil.copy2(BASELINE_MODEL/'reference.jpg', output/'reference.jpg')
    source_videos = {number(p): p for p in (ROOT / 'videos').glob('*.mov')}
    for spec in MULTIVIEW_SPECS:
        dest = output / 'views' / spec['view_id']; dest.mkdir(parents=True, exist_ok=True)
        reference_file = frames_root / spec['reference']
        shutil.copy2(reference_file, dest/'reference.jpg')
        registration = SceneRegistration(cv2.imread(str(reference_file)), spec['anchors'])
        arrays, labels, groups = [], [], []
        for video in spec['videos']:
            files = sorted(frames_root.glob(f'X{video}_*.jpg'))
            sequence = spec['door_labels'][video]
            if len(files) != len(sequence):
                raise ValueError(f'Wrong manually reviewed label count for X{video}: {len(sequence)} vs {len(files)}')
            reviewed = list(zip(files, sequence))
            for t, state in DENSE_DOOR_LABELS.get(video, {}).items():
                reviewed.append((ROOT/'work/states_v2/dense_frames'/f'X{video}_{t:05.1f}.jpg', state))
            digest=sha256(source_videos[video])
            for image, state in reviewed:
                frame = cv2.imread(str(image)); aligned, transform, info = registration.align(frame)
                row = dict(video_number=video, video_sha256=digest,
                    timestamp_sec=float(image.stem.split('_')[1]), image_sha256=sha256(image),
                    door=LABEL[state], safe='not_annotated', split='train', view_id=spec['view_id'], registration=info,
                    annotation_source='assistant_visual_review_saved_full_frames_and_door_contact_sheets')
                annotations.append(row)
                if aligned is None: continue
                x1,y1,x2,y2=spec['rois']['door']
                if np.mean(transform[1][y1:y2,x1:x2] == 255) < .98: continue
                arrays.append(patch_feature(aligned[y1:y2,x1:x2])); labels.append(LABEL[state]); groups.append(video)
        feats=np.asarray(arrays,np.float32); classes=np.asarray(labels,dtype='<U7')
        limits, cross_validation=choose_thresholds(feats,classes,groups)
        cv_reports[spec['view_id']]=cross_validation
        np.savez_compressed(dest/'examples.npz', door_features=feats,door_labels=classes,
                            door_source_videos=np.asarray(groups,dtype=np.int16))
        x1,y1,x2,y2=spec['reflection']
        metadata=dict(schema_version=1,reference_file='reference.jpg',reference_sha256=sha256(dest/'reference.jpg'),
            examples_file='examples.npz',examples_sha256=sha256(dest/'examples.npz'),thresholds={'door':limits},
            rois=spec['rois'],anchor_regions=spec['anchors'],train_videos=spec['videos'],locked_state_supported=False,
            reflection_regions=[dict(region_id='glass_behind_cashier',polygon=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]])])
        atomic_json(dest/'model.json',metadata)
        views.append(dict(view_id=spec['view_id'],model_file=str((dest/'model.json').relative_to(output)).replace('\\','/'),
                          model_sha256=sha256(dest/'model.json')))
    atomic_json(output/'training_annotations.json',dict(frames=annotations,
        note='New poses have only door labels; no safe-state labels are fabricated. X46 screenshot is not a training feature.'))
    atomic_json(output/'training_cross_validation.json',cv_reports)
    atomic_json(output/'model.json',dict(schema_version=2,views=views,
        train_videos=sorted(set(TRAIN) | {v for s in MULTIVIEW_SPECS for v in s['videos']}),
        validation_videos=sorted(set(VALIDATION) | {2,37,46}),
        inference_uses_video_identity=False,classifier='multiview registered local appearances',locked_state_supported=False,
        baseline_model_sha256=sha256(BASELINE_MODEL/'model.json'),
        reflection_policy='Projected regions are context only. A person inside them is not automatically a reflection.'))
    print(json.dumps(dict(views=[v['view_id'] for v in views],counts={s['view_id']:len([r for r in annotations if r['view_id']==s['view_id']]) for s in MULTIVIEW_SPECS}),indent=2))


def evaluate_multiview(output: Path = MULTIVIEW_MODEL):
    cv2.setNumThreads(1)
    output=Path(output)
    metadata=json.loads((output/'model.json').read_text(encoding='utf-8'))
    if set(metadata['train_videos']) & set(metadata['validation_videos']):
        raise ValueError('Multiview train/validation video leakage.')
    baseline=VisualStateDetector(BASELINE_MODEL); detector=VisualStateDetector(output)
    legacy=json.loads((BASELINE_MODEL/'annotations.json').read_text(encoding='utf-8'))
    rows=[]
    for row in legacy['frames']:
        if row['split']=='validation':
            rows.append(dict(video_number=row['video_number'],timestamp_sec=row['timestamp_sec'],
                door=row['door'],safe=row['safe'],source_image=BASELINE_MODEL/row['image'],
                video_sha256=row['video_sha256'],annotation_source='previous_sparse_visual_review'))
    videos={number(p):p for p in (ROOT/'videos').glob('*.mov')}
    for n,times,states in [(2,[0,2,4,6,10,14],'ccoocc'),(37,[0,2,4,6,10,14],'ccoocc'),
                           (46,[0,2,4,6,10,14,17.13],'ccccccc')]:
        digest=sha256(videos[n])
        for t,state in zip(times,states):
            rows.append(dict(video_number=n,timestamp_sec=t,door=LABEL[state],safe='not_annotated',
                source_image=ROOT/'work/states_v2'/f'X{n}_{t:05.2f}.jpg',video_sha256=digest,
                annotation_source='assistant_full_frame_review; user also confirmed X46-like door closed'))
    predictions=[]
    for row in rows:
        source=row.pop('source_image');frame=cv2.imread(str(source))
        if frame is None:raise ValueError(f'Missing reviewed validation frame: {source}')
        dest=output/'reviewed_validation_frames'/f"X{row['video_number']}_{row['timestamp_sec']:06.2f}.jpg"
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        predictions.append(dict(**row,image=str(dest.relative_to(output)),image_sha256=sha256(dest),
                                baseline=baseline.predict(frame),v2=detector.predict(frame)))
    metrics={}
    for model in ['baseline','v2']:
        metrics[model]={}
        for subset in ['legacy_heldout','additional_views','all']:
            selected=[r for r in predictions if subset=='all' or (r['video_number'] in VALIDATION)==(subset=='legacy_heldout')]
            confusion={truth:{pred:0 for pred in ['closed','open','unknown']} for truth in ['closed','open','unknown']}
            for row in selected:confusion[row['door']][row[model]['door']['state']]+=1
            metrics[model][subset]=dict(frames=len(selected),confusion=confusion,
                correct_known=sum(confusion[s][s] for s in ['closed','open']),
                wrong_known=sum(confusion[t][p] for t in confusion for p in ['closed','open'] if t!=p),
                unknown=sum(c['unknown'] for c in confusion.values()))
    report=dict(schema_version=2,model_sha256=sha256(output/'model.json'),
        protocol='Whole-video separation; thresholds use leave-one-training-video-out. Additional-view comparisons are development diagnostics, not an independent deployment accuracy estimate.',
        train_videos=metadata['train_videos'],validation_videos=metadata['validation_videos'],metrics=metrics,predictions=predictions,
        limitations=['Sparse frame labels do not establish complete event recall or timing accuracy.',
            'New camera views currently have only door-state references. Safe states remain unknown there.',
            'The front white-shirt view contains only closed-door examples; unseen open configurations may abstain.',
            'Reflection regions are spatial context only, never a blanket exclusion of doorway people.',
            'User-provided cropped screenshot was reviewed as a closed-door example and was not used as an inference feature.'])
    atomic_json(output/'validation_report.json',report)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--evaluate-only',action='store_true');p.add_argument('--multiview',action='store_true')
    args=p.parse_args()
    if args.multiview:
        output=args.output or MULTIVIEW_MODEL
        if args.evaluate_only:evaluate_multiview(output)
        else:build_multiview(output)
    else:
        output=args.output or BASELINE_MODEL
        if not args.evaluate_only:build(output)
        evaluate(output)
