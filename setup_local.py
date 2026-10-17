"""Create a private Python environment without modifying global packages."""
from pathlib import Path
import argparse,subprocess,sys,venv
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--cuda',action='store_true',help='Install CUDA 12.1 PyTorch wheels for NVIDIA');args=p.parse_args()
    if sys.version_info[:2]!=(3,11):raise SystemExit('Use Python 3.11: py -3.11 setup_local.py')
    from check_install import verify_weights
    verify_weights()
    env=ROOT/'.venv'
    if not env.exists():venv.EnvBuilder(with_pip=True).create(env)
    python=env/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    def run(*parts):subprocess.run([str(python),*parts],cwd=ROOT,check=True)
    run('-m','pip','install','--upgrade','pip')
    variant='cu121' if args.cuda else 'cpu'
    run('-m','pip','install','torch==2.5.1+'+variant,'torchvision==0.20.1+'+variant,'--index-url',
        'https://download.pytorch.org/whl/'+('cu121' if args.cuda else 'cpu'),'--extra-index-url','https://pypi.org/simple')
    run('-m','pip','install','-r','requirements.txt')
    run('check_install.py')
    print('Ready. Windows: .venv\\Scripts\\python.exe launch_web.py')
if __name__=='__main__':main()
