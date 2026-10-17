"""Create, validate, import, and evaluate private local human annotations.

Only aggregate counts are printed. No model is called and no network is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.annotations import (AnnotationError, annotation_template, import_annotation,
                                inspect_source, load_annotation, proposal_metrics,
                                split_group_ids, validate_annotation, write_exclusive_json)
from worker.video import AnalysisError


def catalog_types() -> set[str]:
    catalog = json.loads((ROOT / "rules" / "catalog.json").read_text(encoding="utf-8"))
    return {item["id"] for item in catalog["items"]} | {"other_observable_event", "unknown"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Write an empty draft and exact local source PTS")
    init.add_argument("--video", required=True, type=Path)
    init.add_argument("--output", required=True, type=Path)
    for command in ("validate", "import"):
        command_parser = commands.add_parser(command)
        command_parser.add_argument("--video", required=True, type=Path)
        command_parser.add_argument("--annotation", required=True, type=Path)
        if command == "import":
            command_parser.add_argument("--attest-reviewed", required=True, metavar="REVIEWER",
                                        help="Explicit human review attestation; must match the file reviewer")
            command_parser.add_argument("--directory", type=Path, default=ROOT / "data" / "annotations")
    evaluate = commands.add_parser("evaluate", help="Evaluate reviewed imports against local prediction JSON")
    evaluate.add_argument("--directory", type=Path, default=ROOT / "data" / "annotations")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--sources", type=Path, required=True,
                          help="Private JSON with sources: [{video_sha256, video_path}] for source revalidation")
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            annotations = [load_annotation(path) for path in sorted(args.directory.glob("*.json"))]
            sources = load_annotation(args.sources)
            if set(sources) != {"sources"} or not isinstance(sources["sources"], list):
                raise AnnotationError("Evaluation needs explicit local source paths")
            by_hash = {}
            for source in sources["sources"]:
                if set(source) != {"video_sha256", "video_path"} or source["video_sha256"] in by_hash:
                    raise AnnotationError("Evaluation source hashes must be unique")
                by_hash[source["video_sha256"]] = source["video_path"]
            for annotation in annotations:
                digest = annotation["source"]["sha256"]
                info, verified_digest = inspect_source(by_hash[digest])
                validate_annotation(annotation, info, verified_digest, require_reviewed=True,
                                    allowed_types=catalog_types())
            predictions = load_annotation(args.predictions)
            if set(predictions) != {"predictions"} or not isinstance(predictions["predictions"], list):
                raise AnnotationError("Evaluation input needs a predictions list")
            result = proposal_metrics(annotations, predictions["predictions"])
            result["split_components"] = len(set(split_group_ids(annotations).values()))
            write_exclusive_json(args.output, result)
            print(json.dumps({"status": "evaluated", "matched_sources": result["matched_sources"],
                              "missing_predictions": result["missing_predictions"]}))
            return 0
        info, digest = inspect_source(args.video)
        if args.command == "init":
            annotation = annotation_template(info, digest)
            write_exclusive_json(args.output, annotation)
            # Private review aid beside the template. Source indices, PTS and
            # duration contain no invented labels and never enter inference.
            write_exclusive_json(args.output.with_suffix(".pts.json"),
                                 {"source_sha256": digest, "duration_sec": info.duration,
                                  "frames": [{"source_index": i, "timestamp_sec": pts}
                                             for i, pts in enumerate(info.pts)]})
            print(json.dumps({"status": "draft_created", "events": 0, "source_frames": len(info.pts)}))
            return 0
        annotation = load_annotation(args.annotation)
        summary = validate_annotation(annotation, info, digest, allowed_types=catalog_types())
        if args.command == "import":
            import_annotation(annotation, info, digest, args.directory,
                              attesting_reviewer=args.attest_reviewed, allowed_types=catalog_types())
        print(json.dumps({"status": "imported" if args.command == "import" else "valid", **summary}))
        return 0
    except (AnnotationError, AnalysisError, OSError, ValueError, TypeError, KeyError):
        # Exception strings can contain paths or user annotation content. Keep
        # stdout/stderr aggregate-only; inspect the local file to correct it.
        print(json.dumps({"status": "rejected", "reason": "Invalid local source, schema, review attestation, or destination conflict"}),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
