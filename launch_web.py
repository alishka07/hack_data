"""Start the local video-review page using this global Python interpreter."""
from pathlib import Path
import argparse
import json
import os
import socket
import threading
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('port must be between 1024 and 65535')
    url = f'http://127.0.0.1:{args.port}'
    try:
        with urlopen(url+'/api/health', timeout=1) as response:
            existing = json.load(response)
        if existing.get('app') == 'fortebank-local-video-review':
            print(f'Already running: {url}', flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            return
    except Exception:
        pass
    with socket.socket() as check:
        try:
            check.bind(('127.0.0.1', args.port))
        except OSError:
            parser.error(f'Port {args.port} is in use. Try --port {args.port+1}.')
    if not args.no_browser:
        def open_when_ready():
            for _ in range(40):
                try:
                    with urlopen(url+'/api/health', timeout=.5) as response:
                        if json.load(response).get('app') == 'fortebank-local-video-review':
                            webbrowser.open(url)
                            return
                except Exception:
                    time.sleep(.25)
        threading.Thread(target=open_when_ready, daemon=True).start()
    os.chdir(ROOT)
    import uvicorn
    print(f'Video review: {url}\nKeep this process running while using the page.', flush=True)
    uvicorn.run('web_app:app', host='127.0.0.1', port=args.port, log_level='info', access_log=False)


if __name__ == '__main__':
    main()
