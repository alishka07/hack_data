"""Mark fixed objects/exclusions in one video. Never silently reuse across videos."""
import argparse
import json
from pathlib import Path
import cv2
from run import resolve_video

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=ROOT / 'videos' / 'X-1.mov')
    parser.add_argument('--second', type=float, default=10)
    parser.add_argument('--labels', nargs='+', default=['banknote_counter', 'safe', 'laptop', 'exclude'])
    args = parser.parse_args()
    video = resolve_video(args.video)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.second * 1000)
    ok, frame = cap.read()
    pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    cap.release()
    if not ok:
        raise SystemExit('Cannot read the requested frame. Try --second 0.')
    height, width = frame.shape[:2]
    regions = []
    print('Draw rectangle, then ENTER. Press C to skip. exclude = outlet/power strip; avoid selecting the whole desk.')
    for label in args.labels:
        window = f'{label}: rectangle + ENTER; C = skip'
        x, y, w, h = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window)
        if w and h:
            regions.append(dict(kind='exclude' if label == 'exclude' else 'equipment', label='banknote_counter' if label == 'exclude' else label,
                                box=[x / width, y / height, (x + w) / width, (y + h) / height], source='manual'))
    if not regions:
        print('No regions selected; existing calibration was not changed.')
        return
    output = ROOT / 'scenes' / (video.stem + '.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(dict(video_name=video.name, video_size_bytes=video.stat().st_size,
                      image_size=[width, height], reference_timestamp_sec=pts, regions=regions,
                      note='Manual hints for this video. Not model detections. Recalibrate if the camera or equipment moves.'),
                      indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
