"""Make complete browser playback copies while preserving original recordings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import cv2
import imageio_ffmpeg
from annotation_api import atomic_json


def prepare(source, root):
    number = re.search(r'\d+', source.stem).group()
    video_id = 'X'+str(int(number))
    out = root/'previews'/f'{video_id}.mp4'
    proof = out.with_suffix('.json')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if out.exists() and proof.exists() and json.loads(proof.read_text())['source_sha256'] == source_hash:
        return json.loads(proof.read_text())
    temp = out.with_name(out.stem+'.part.mp4')
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-hide_banner', '-loglevel', 'error', '-i', str(source),
               '-map', '0:v:0', '-map', '0:a?', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough', '-c:a', 'aac', '-movflags', '+faststart',
               '-threads', '2', str(temp)]
    result = subprocess.run(command, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    counts = []
    for path in (source, temp):
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        count, first, last = 0, None, None
        while cap.grab():
            t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            first = t if first is None else first
            last = t
            count += 1
        cap.release()
        counts.append(dict(frames=count, first_pts=first, last_pts=last))
    if counts[0]['frames'] != counts[1]['frames'] or not counts[0]['frames']:
        raise RuntimeError(f'Preview truncated or frame count changed: {video_id}, {counts}')
    if abs(counts[0]['last_pts']-counts[1]['last_pts']) > .1:
        raise RuntimeError(f'Preview timeline changed: {video_id}, {counts}')
    temp.replace(out)
    data = dict(video_id=video_id, source_sha256=source_hash, source=counts[0], preview=counts[1],
                playback_copy='H264_MP4_full_recording', audio='preserved_if_present_as_AAC',
                original_modified=False, elapsed_sec=round(time.perf_counter()-started,3))
    atomic_json(proof, data)
    print(f'{video_id}: {counts[0]["frames"]} frames verified', flush=True)
    return data


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--videos',type=Path,default=Path(__file__).parent/'videos')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'annotations/video_review')
    args=parser.parse_args()
    sources=sorted(args.videos.glob('*.mov'),key=lambda p:int(re.search(r'\d+',p.stem).group()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(lambda path:prepare(path,args.output),sources))
    atomic_json(args.output/'preview_verification.json',{'videos':rows,'verified_full_recordings':len(rows)})
