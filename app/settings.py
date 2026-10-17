"""Shared configuration. All processing and storage default to this machine."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import secrets
import time


@dataclass
class Settings:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path | None = None
    db_path: Path | None = None
    api_key: str = ""
    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_bytes: int = 128 * 1024 * 1024
    job_timeout_sec: int = 300
    worker_stale_sec: int = 30
    ffprobe_path: str = "ffprobe"

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.data_dir = Path(self.data_dir or self.root / "data").resolve()
        self.db_path = Path(self.db_path or self.data_dir / "jobs.sqlite").resolve()
        if self.max_upload_bytes < 1 or self.job_timeout_sec < 1 or self.worker_stale_sec < 1:
            raise ValueError("Upload, processing, and heartbeat limits must be positive")
        if not 1 <= self.port <= 65535:
            raise ValueError("FORTE_PORT must be a valid TCP port")


def _private_api_key(data_dir: Path) -> str:
    """Persist an unpredictable key atomically; never print it or send it to a client."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "api-key.txt"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # API and worker may first start together. Let the winning creator finish
        # writing its very small file before treating an empty key as corruption.
        value = ""
        for attempt in range(10):
            value = path.read_text(encoding="utf-8").strip()
            if value:
                break
            time.sleep(0.02)
        if not value:
            raise ValueError("The saved API key is empty; set FORTE_API_KEY or restore api-key.txt")
        return value
    value = secrets.token_urlsafe(32)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(value + "\n")
        output.flush()
        os.fsync(output.fileno())
    return value


def load_settings() -> Settings:
    root = Path(os.environ.get("FORTE_ROOT", Path(__file__).resolve().parents[1]))
    data_dir = Path(os.environ.get("FORTE_DATA_DIR", root / "data"))
    key = os.environ.get("FORTE_API_KEY", "").strip() or _private_api_key(data_dir)
    return Settings(
        root=root,
        data_dir=data_dir,
        db_path=Path(os.environ.get("FORTE_DB_PATH", data_dir / "jobs.sqlite")),
        api_key=key,
        host=os.environ.get("FORTE_HOST", "127.0.0.1"),
        port=int(os.environ.get("FORTE_PORT", "8000")),
        max_upload_bytes=int(os.environ.get("FORTE_MAX_UPLOAD_BYTES", str(128 * 1024 * 1024))),
        job_timeout_sec=int(os.environ.get("FORTE_JOB_TIMEOUT_SEC", "300")),
        worker_stale_sec=int(os.environ.get("FORTE_WORKER_STALE_SEC", "30")),
        ffprobe_path=os.environ.get("FORTE_FFPROBE", "ffprobe"),
    )
