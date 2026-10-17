"""Fit version-bound decision cutoffs from reviewed verifier candidate outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.fusion import fit_thresholds
from worker.pipeline import PIPELINE_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON array or JSONL of reviewed candidate outcomes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True, help="Exact runtime metadata model_id")
    parser.add_argument("--target-precision", type=float, default=.9)
    parser.add_argument("--minimum-examples", type=int, default=20, help="Per rule and partition")
    parser.add_argument("--minimum-class", type=int, default=5)
    args = parser.parse_args()
    raw = args.input.read_text(encoding="utf-8")
    rows = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(line) for line in raw.splitlines() if line.strip()]
    artifact = fit_thresholds(rows, model_id=args.model_id, pipeline_version=PIPELINE_VERSION,
                             target_precision=args.target_precision, minimum_examples=args.minimum_examples,
                             minimum_class=args.minimum_class)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rules": {key: value["status"] for key, value in artifact["rules"].items()}}))


if __name__ == "__main__":
    main()
