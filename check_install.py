"""Verify weights, dependencies and self-contained model imports before launch."""
from pathlib import Path
import argparse,hashlib,importlib,json,sys
ROOT=Path(__file__).resolve().parent

def verify_weights(root=ROOT):
    manifest=json.loads((root/'models/manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path=root/row['path']
        if not path.is_file() or path.stat().st_size!=row['bytes']:
            raise RuntimeError(f'Missing or incomplete weight: {row["path"]}. Run git lfs pull.')
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=row['sha256']:raise RuntimeError(f'Checksum mismatch: {row["path"]}. Run git lfs pull.')
    return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights-only',action='store_true');args=p.parse_args()
    manifest=verify_weights()
    print(f'Weights verified: {manifest["total_bytes"]:,} bytes')
    if args.weights_only:return
    for name in ('torch','torchvision','cv2','transformers','fastapi','imageio_ffmpeg','lap','faster_coco_eval'):
        module=importlib.import_module(name);print(name,getattr(module,'__version__','OK'))
    from train_objects import REPO
    sys.path.insert(0,str(REPO));importlib.import_module('run_local')
    sys.path.insert(0,str(ROOT/'runtime/ByteTrack-main'));importlib.import_module('run_rtdetr_bytetrack')
    sys.path.insert(0,str(ROOT/'runtime/GroundingDINO-local'));importlib.import_module('detect_local')
    print('Local adapters OK. Start: python launch_web.py')

if __name__=='__main__':main()
