"""Bind browser timestamps to source frame indices, including negative preroll."""
import json
import hashlib
from pathlib import Path
import cv2
from annotation_api import atomic_json

root=Path(__file__).parent/'annotations/video_review'
for preview in sorted((root/'previews').glob('X*.mp4')):
    proof=json.loads(preview.with_suffix('.json').read_text(encoding='utf-8'))
    cached=preview.with_suffix('.timeline.json')
    preview_sha=hashlib.sha256(preview.read_bytes()).hexdigest()
    old=json.loads(cached.read_text(encoding='utf-8')) if cached.is_file() else {}
    timestamps=old.get('timestamps',[]) if old.get('preview_sha256')==preview_sha and old.get('source_sha256')==proof['source_sha256'] else []
    if len(timestamps)!=proof['source']['frames']:
        cap=cv2.VideoCapture(str(preview),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
        timestamps=[]
        while cap.grab():timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        cap.release()
    if len(timestamps)!=proof['source']['frames']:raise ValueError(f'Frame count changed: {preview}')
    ledger=root/'videos'/preview.stem/'frames.jsonl'
    atomic_json(preview.with_suffix('.timeline.json'),{'source_sha256':proof['source_sha256'],'timestamps':timestamps,
                'preview_sha256':preview_sha,
                'source_frame_ledger_sha256':hashlib.sha256(ledger.read_bytes()).hexdigest(),
                'mapping':'same_frame_index_after_full_count_verification','verified':True})
print('Playback frame timelines indexed',flush=True)
