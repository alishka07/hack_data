"""Run the locally fine-tuned four-class RT-DETR detector on recorded videos."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import run as report_runner

ROOT = Path(__file__).resolve().parent


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--video',type=Path)
    selection.add_argument('--all',action='store_true')
    parser.add_argument('--videos',type=Path,default=ROOT/'videos')
    parser.add_argument('--weights',type=Path,default=ROOT/'training/rtdetr_custom_final/best.pth')
    parser.add_argument('--output',type=Path,default=ROOT/'results/trained_objects')
    parser.add_argument('--interval',type=float,default=1.0)
    parser.add_argument('--threshold',type=float,default=None,help='Override all fitted class thresholds; default uses calibration.json')
    parser.add_argument('--device',default=None)
    parser.add_argument('--max-samples',type=int,default=0)
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval<.04 or args.max_samples<0:
        parser.error('interval must be finite and >=.04, max-samples >=0')
    if args.threshold is not None and not 0<args.threshold<1:
        parser.error('threshold must be in (0,1)')
    if not args.weights.is_file():
        parser.error(f'Train the model first; checkpoint missing: {args.weights}')
    videos = [report_runner.resolve_video(args.video,args.videos)] if args.video else report_runner.discover(args.videos)
    if not videos:
        parser.error('No input videos')
    checkpoint_hash = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    calibration_path = args.weights.with_name('calibration.json')
    thresholds = {}
    if args.threshold is None and calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
        if calibration['checkpoint_sha256'] != checkpoint_hash:
            parser.error('calibration.json belongs to different model weights. Re-run calibrate_threshold.py.')
        thresholds = calibration['thresholds']
    if args.threshold is None and not thresholds:
        print('No fitted thresholds found; using 0.35. Set --threshold if needed.',flush=True)
        args.threshold = .35
    provenance = {'engine':'fine_tuned_rtdetrv2', 'checkpoint_sha256':checkpoint_hash,
                  'checkpoint':str(args.weights.resolve()),'per_class_thresholds':thresholds,
                  'global_threshold_override':args.threshold}
    # The reusable report pipeline records this in the result signature.
    report_runner.PROMPTS['trained'] = json.dumps(provenance,sort_keys=True)
    args.groups = ['trained']
    args.output = args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    print('Loading locally trained RT-DETR: safe, counter, laptop, bag...',flush=True)
    from train_objects import load_trained,predict_image
    model = load_trained(args.weights,args.device)
    loaded = (model,None,str(model[3]))

    def detect(image,wrapped,unused_prompt,threshold):
        cutoff = threshold if threshold is not None else min(thresholds.values())
        predictions = predict_image(image,wrapped[0],threshold=cutoff)
        return [row for row in predictions if row['score'] >= (threshold if threshold is not None else thresholds.get(row['label'],.35))]

    rows = []
    for video in videos:
        try:
            row = report_runner.analyze(video,args,loaded,detect)
            row['model_provenance'] = provenance
            row['training_scope'] = 'Small staged local dataset; object recognition only, no automatic violation verdict.'
            report_runner.write_json(args.output/video.stem/'summary.json',row)
            rows.append(row)
        except Exception as error:
            import traceback
            traceback.print_exc()
            rows.append({'status':'error','video':str(video),'error':str(error)})
        report_runner.write_json(args.output/'batch_summary.json',rows)
        report_runner.make_index(args.output,rows)
    errors = sum(row['status']=='error' for row in rows)
    print(f'Finished {len(rows)-errors}/{len(rows)} videos. Open {args.output/"index.html"}',flush=True)
    return int(errors>0)


if __name__=='__main__':
    raise SystemExit(main())
