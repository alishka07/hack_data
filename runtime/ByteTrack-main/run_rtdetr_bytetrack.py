"""Run the downloaded ByteTrack with RT-DETRv2 detections (people only)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'RT-DETR-main' / 'rtdetrv2_pytorch'))

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, ToTensor
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.basetrack import BaseTrack
from run_local import load_model


@torch.inference_mode()
def run(video, output, start=0.0, duration=5.0):
    video, output = Path(video), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    # Bound decoder threads as well as Torch threads on machines with little RAM.
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise RuntimeError('Video has no usable frame rate')
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    model, postprocessor, device = load_model()
    print(f'RT-DETR loaded on {device}; ByteTrack source: {ROOT}', flush=True)
    BaseTrack._count = 0
    tracker = BYTETracker(SimpleNamespace(track_thresh=0.5, track_buffer=30,
                                         match_thresh=0.8, mot20=False), frame_rate=fps)
    transform = Compose([Resize((640, 640)), ToTensor()])
    size = torch.tensor([[width, height]], device=device)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Cannot create output video')
    count = 0
    csv_rows = 0
    ids = set()
    first_pts = None
    last_pts = None
    timestamp_normalizations = 0
    source_timeline = []
    started = time.perf_counter()
    try:
        with output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
            csvout = csv.writer(stream)
            csvout.writerow(['source_frame', 'timestamp_sec', 'track_id', 'class', 'score', 'x1', 'y1', 'x2', 'y2'])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                source_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if not np.isfinite(pts) or pts < 0:
                    pts = max(0.0, source_frame / fps)
                    timestamp_normalizations += 1
                if pts < start:
                    continue
                if duration > 0 and pts >= start + duration:
                    break
                first_pts = pts if first_pts is None else first_pts
                last_pts = pts
                source_timeline.append([pts, count / fps])
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor = transform(image).unsqueeze(0).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    labels, boxes, scores = postprocessor(model(tensor), size)
                # Keep low-confidence detections for ByteTrack's second association.
                keep = (labels[0] == 0) & (scores[0] > 0.1)
                dets = torch.cat((boxes[0][keep], scores[0][keep, None]), dim=1).cpu().numpy()
                dets = np.ascontiguousarray(dets, dtype=np.float64).reshape(-1, 5)
                # RT-DETR already returns original-image coordinates: use scale 1.
                tracks = tracker.update(dets, (height, width), (height, width))
                for track in tracks:
                    coords = track.tlbr
                    if not np.isfinite(coords).all():
                        continue
                    tid = int(track.track_id)
                    ids.add(tid)
                    csvout.writerow([source_frame, pts, tid,
                                     'person', float(track.score), *coords.tolist()])
                    csv_rows += 1
                    x1, y1, x2, y2 = np.rint(coords).astype(int)
                    color = ((tid * 67) % 180 + 60, (tid * 41) % 180 + 60, (tid * 97) % 180 + 60)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'person ID {tid}', (max(0, x1), max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                cv2.putText(frame, f'{pts:.2f}s | RT-DETR + ByteTrack', (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                writer.write(frame)
                count += 1
                if count == 1 or count % 30 == 0:
                    cv2.imwrite(str(output.with_suffix('.jpg')), frame)
                    print(f'{count} frames; time={pts:.2f}s; active tracks={len(tracks)}', flush=True)
    finally:
        cap.release()
        writer.release()
    if count == 0:
        raise RuntimeError('No frames processed: check start and duration')
    # OpenCV's mp4v is not supported by many browser players; export H.264.
    playable = output.with_name(output.stem + '.encoding.mp4')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-hide_banner',
                    '-loglevel', 'error', '-i', str(output), '-an', '-c:v',
                    'libx264', '-threads', '2', '-preset', 'fast', '-crf', '20', '-pix_fmt',
                    'yuv420p', '-movflags', '+faststart', str(playable)], check=True)
    playable.replace(output)
    full_video_requested = start == 0 and duration == 0
    frame_count_matches = (count == expected_frames) if full_video_requested and expected_frames > 0 else None
    decoding_warning = 'Only part of the source video was processed; inspect source/decoder memory errors.' if frame_count_matches is False else None
    report = dict(video=str(video), output=str(output), device=str(device), frames=count,
                  source_fps=fps, first_timestamp_sec=first_pts, last_timestamp_sec=last_pts,
                  metadata_frames=expected_frames, full_video_requested=full_video_requested,
                  frame_count_matches=frame_count_matches, decoding_warning=decoding_warning,
                  timestamp_normalizations=timestamp_normalizations, decoder_threads=1, torch_threads=2,
                  source_timeline=source_timeline,
                  cuda_mixed_precision=device.type == 'cuda',
                  csv_rows=csv_rows, csv_sha256=hashlib.sha256(output.with_suffix('.csv').read_bytes()).hexdigest(),
                  track_ids=sorted(ids), processing_seconds=time.perf_counter() - started,
                  note='Temporary track IDs, not identities or verified unique-person counts.')
    output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--duration', type=float, default=5)
    args = parser.parse_args()
    run(args.video, args.output, args.start, args.duration)
