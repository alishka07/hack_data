# Share a temporary jury demo

The jury can use the website in a browser with an API key, without installing models or Python. Automated clients use the same public host with `POST /api/v1/analyze` and `GET /api/v1/jobs/{job_id}`, sending the key in `X-API-Key`.

The demo launcher creates a separate `data/jury/` directory for its key, database, uploads, evidence and reviews. It does not publish the ordinary app's saved analyses. Everyone given the jury key shares the jury history and review permissions.

## Start on the prepared laptop

First start the normal local service and wait for its model server at `http://127.0.0.1:18080/health` to report `ok`. Install the official [Cloudflare tunnel executable](https://developers.cloudflare.com/tunnel/downloads/) separately. From the repository directory:

```powershell
.\.venv\Scripts\python.exe scripts/run_jury_demo.py --cloudflared C:\tools\cloudflared.exe
```

The launcher starts a separate loopback API on port 8001 and a worker that attaches to the already loaded model. It starts a temporary HTTPS tunnel to port 8001. It never exposes the model port or prints the API key.

After the launcher reports the public address, share:

- Website: the address in `data/jury/public-url.txt`, followed by `/`.
- API base: the same address followed by `/api/v1`.
- API key: the value in `data/jury/api-key.txt`, shared privately with the jury.

The browser asks for the key before uploading. Uploads accept MOV files up to 95,000,000 bytes to leave room for multipart overhead below the tunnel provider's 100 MB request limit. Analysis runs one video at a time. API clients upload with multipart field `file`, receive a job ID, then poll for the result.

## Availability

Keep the laptop powered, online, and the normal service and demo launcher running. On Windows the demo launcher requests that the system stay awake while it is running; this does not prevent a shutdown or every lid-close policy. Stop the demo with Ctrl+C in its terminal. Only demo-owned processes are stopped; the existing model server and original service are retained.

The two app instances share one GPU model. Do not submit analyses through the ordinary localhost:8000 app while the jury is testing. Their databases do not provide shared admission control.

The [temporary Cloudflare URL](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/) changes when the tunnel is restarted and has no uptime guarantee. For a fixed submission address, configure a named tunnel and domain or a hosted GPU service. A GitHub repository URL does not host the inference service.

Keep `data/` and tunnel binaries out of Git. Private videos, existing API keys and local runtime paths must not be included in a public handoff. Test the public website, unauthorized rejection, isolated history and one complete upload before sending the address.
