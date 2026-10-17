"""Start the local application using this laptop's bootstrap configuration."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config = ROOT / "data/local-runtime.json"
    python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not config.is_file() or not python.is_file():
        print("Run python scripts/bootstrap_local.py first; see docs/local-setup.md", file=sys.stderr)
        return 2
    try:
        values = json.loads(config.read_text(encoding="utf-8"))
        if not isinstance(values, dict) or any(not isinstance(k, str) or not k.startswith("FORTE_")
                                               or not isinstance(v, str) for k, v in values.items()):
            raise ValueError("Local runtime configuration must contain FORTE_* string settings")
        env = os.environ.copy()
        # Explicit shell settings override the generated machine defaults.
        for key, value in values.items():
            env.setdefault(key, value)
        for key in ("FORTE_LLAMA_SERVER", "FORTE_MODEL_PATH", "FORTE_MMPROJ_PATH", "FORTE_OBJECT_PYTHON"):
            if not Path(env.get(key, "")).is_file():
                raise ValueError(f"{key} is missing; rerun bootstrap or set its path explicitly")
        return subprocess.call([str(python), str(ROOT / "scripts/run_local.py"), *sys.argv[1:]], cwd=ROOT, env=env)
    except (OSError, ValueError) as exc:
        print(f"Startup stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
