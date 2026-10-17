"""Reapply current filters/manual regions to saved raw predictions, without inference."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from candidate_filters import filter_candidates
from run import ROOT, RENDER_FOOTER, VERSION, draw_frame, make_index, read_regions, write_json
from reviewed_objects import ReviewedObjectMatcher, DEFAULT_ANNOTATIONS


def process(folder):
    summary_path = folder / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    video = Path(summary['video'])
    if video.stat().st_size != summary['signature']['video_size'] or video.stat().st_mtime_ns != summary['signature']['video_mtime']:
        raise RuntimeError(f'The input video changed: {video}')
    width, height = summary['image_size']
    regions, calibration = read_regions(video, width, height)
    records = [json.loads(line) for line in (folder / 'frames.jsonl').read_text(encoding='utf-8').splitlines() if line]
    groups = summary['signature'].get('groups', [])
    reviewed_enabled = bool({'cash', 'food_drink'}.intersection(groups))
    reviewed_matcher = ReviewedObjectMatcher.for_video(video) if reviewed_enabled else None
    counts = Counter()
    cap = cv2.VideoCapture(str(video))
    writer = None
    if summary['video_preview']:
        import imageio_ffmpeg
        writer = imageio_ffmpeg.write_frames(str(folder / 'review.refilter.mp4'), (width, height + RENDER_FOOTER),
                 fps=max(.1, 1 / summary['sample_interval_sec']), codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
                 macro_block_size=1, ffmpeg_log_level='error',
                 output_params=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-movflags', '+faststart', '-crf', '21'])
        writer.send(None)
    done = set()
    targets = {r['source_frame']: (i, r) for i, r in enumerate(records)}
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index in targets:
                ordinal, record = targets[index]
                raw = record['raw_detections']
                effective, corrections = reviewed_matcher.apply(frame, raw) if reviewed_matcher else (raw, [])
                result = filter_candidates(effective, (width, height), regions)
                record.update(result)
                record['reviewed_object_corrections'] = corrections
                for status, rows in result.items():
                    counts[status] += len(rows)
                rendered = draw_frame(frame, record, record['timestamp_sec'], summary['sample_interval_sec'])
                if writer is not None:
                    writer.send(np.ascontiguousarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)))
                if ordinal < len(summary['thumbnails']):
                    cv2.imencode('.jpg', rendered, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(folder / summary['thumbnails'][ordinal]['file']))
                done.add(index)
            index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.close()
    if done != set(targets):
        raise RuntimeError(f'Could not decode all saved sample frames in {video}')
    if writer is not None:
        (folder / 'review.refilter.mp4').replace(folder / 'review.mp4')
    with (folder / 'frames.jsonl').open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    with (folder / 'detections.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        csvout = csv.writer(stream)
        csvout.writerow(['timestamp_sec', 'source_frame', 'status', 'label', 'score', 'x1', 'y1', 'x2', 'y2', 'reasons', 'source', 'score_kind', 'reference_frame'])
        for record in records:
            for status in ('candidates', 'review', 'suppressed'):
                for row in record[status]:
                    csvout.writerow([record['timestamp_sec'], record['source_frame'], status, row.get('label', ''), row.get('score'), *(row.get('box') or [None]*4), '|'.join(row.get('reasons', [])), row.get('source', 'model'), row.get('score_kind', 'model_score'), row.get('reference_frame_zero_based')])
    summary['counts'] = dict(counts)
    summary['signature']['version'] = VERSION
    summary['calibration'] = calibration
    summary['signature']['filter_hash'] = hashlib.sha256((ROOT / 'candidate_filters.py').read_bytes()).hexdigest()
    summary['signature']['reviewed_objects_hash'] = hashlib.sha256((ROOT / 'reviewed_objects.py').read_bytes()).hexdigest()
    summary['signature']['reviewed_annotations_hash'] = hashlib.sha256(DEFAULT_ANNOTATIONS.read_bytes()).hexdigest() if DEFAULT_ANNOTATIONS.is_file() and reviewed_enabled else None
    summary['reviewed_object_reference_enabled'] = reviewed_matcher is not None
    summary['reviewed_reference_scope'] = 'exact_source_sha256_only'
    scene_path = ROOT / 'scenes' / f'{video.stem}.json'
    summary['signature']['scene_hash'] = hashlib.sha256(scene_path.read_bytes()).hexdigest() if scene_path.exists() else None
    summary['refiltered_at_unix'] = time.time()
    summary['refilter_note'] = 'Saved raw detections re-filtered; model inference not repeated.'
    write_json(summary_path, summary)
    print(f'REFILTERED {video.name}: {len(records)} samples', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results' / 'all_videos_5s')
    args = parser.parse_args()
    rows = []
    for path in sorted(args.results.glob('*/summary.json')):
        rows.append(process(path.parent))
    if not rows:
        parser.error('No completed saved results found. Run run.py first.')
    write_json(args.results / 'batch_summary.json', rows)
    make_index(args.results, rows)
    print(f'Rebuilt {args.results / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
