"""Use the existing RT-DETR + ByteTrack runner from the case folder."""
import argparse
import math
import os
from pathlib import Path
import sys
from run import resolve_video

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=ROOT / 'videos' / 'X-1.mov')
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--duration', type=float, default=0, help='0 means the complete video')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.start) or not math.isfinite(args.duration) or args.start < 0 or args.duration < 0:
        parser.error('start and duration must be finite and non-negative')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    video = resolve_video(args.video)
    output = args.output or ROOT / 'results' / 'people' / (video.stem + '.mp4')
    sys.path.insert(0, str(ROOT / 'runtime' / 'ByteTrack-main'))
    from run_rtdetr_bytetrack import run
    run(video, output, start=args.start, duration=args.duration)


if __name__ == '__main__':
    main()
