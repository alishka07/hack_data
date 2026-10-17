"""Fresh subprocess for Jupyter/global kernel: %run start.py --video X-1.mov"""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    arguments = sys.argv[1:]
    script = 'run.py'
    if '--trained' in arguments:
        arguments.remove('--trained')
        script = 'detect_trained.py'
    raise SystemExit(subprocess.call([sys.executable, '-u', str(Path(__file__).with_name(script)), *arguments]))
