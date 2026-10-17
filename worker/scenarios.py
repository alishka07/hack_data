"""Fixed visual review coverage; runtime never reads example labels or filenames.

Visual support establishes an observable concern for review. It does not replace
the separate legal registry's applicability, exception or authorization checks.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re


_ID = re.compile(r"[a-z][a-z0-9_]{0,95}\Z")
_FIELDS = {"id", "label", "visual_checks", "check_descriptions", "visual_cues", "unknowns",
           "related_legal_rule_ids", "legal_source", "disposition", "alias_of"}


def load_scenarios(root: Path) -> dict[str, dict]:
    """Read only the curated, immutable inference taxonomy under ``rules``."""
    value = json.loads((root / "rules" / "scenarios.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("version"), str):
        raise ValueError("The visual scenario registry needs a version")
    rows = value.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError("The visual scenario registry needs scenario rows")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != _FIELDS:
            raise ValueError("A visual scenario has invalid fields")
        identifier = row["id"]
        if not isinstance(identifier, str) or not _ID.fullmatch(identifier) or identifier in result:
            raise ValueError("Visual scenario IDs must be unique identifiers")
        if not isinstance(row["label"], str) or not row["label"].strip():
            raise ValueError("A visual scenario needs a label")
        for field in ("visual_checks", "visual_cues", "unknowns", "related_legal_rule_ids"):
            if (not isinstance(row[field], list) or not row[field]
                    or any(not isinstance(item, str) or not item.strip() for item in row[field])
                    or len(row[field]) != len(set(row[field]))):
                # An explicitly unmapped source may have no legal rule IDs.
                if field != "related_legal_rule_ids" or row[field] != []:
                    raise ValueError("A visual scenario needs unique, nonempty check and cue lists")
        descriptions = row["check_descriptions"]
        if (any(not _ID.fullmatch(key) for key in row["visual_checks"])
                or not isinstance(descriptions, dict) or set(descriptions) != set(row["visual_checks"])
                or any(not isinstance(text, str) or not text.strip() for text in descriptions.values())):
            raise ValueError("Every visual check needs a matching description")
        source = row["legal_source"]
        if source is not None and (not isinstance(source, dict)
                or set(source) != {"document", "paragraph", "source_url"}
                or any(not isinstance(text, str) or not text.strip() for text in source.values())):
            raise ValueError("A mapped legal source needs its actual document, paragraph and URL")
        if row["disposition"] != "review_only":
            raise ValueError("Visual scenarios cannot expand automatic legal findings")
        if row["alias_of"] is not None and (not isinstance(row["alias_of"], str)
                                            or not _ID.fullmatch(row["alias_of"])):
            raise ValueError("A visual scenario alias must name another scenario")
        result[identifier] = deepcopy(row)
    for identifier, row in result.items():
        alias = row["alias_of"]
        if alias is not None and (alias not in result or alias == identifier or result[alias]["alias_of"] is not None):
            raise ValueError("Visual aliases must name one canonical scenario without cycles")
    return result


def scenario_type_ids(scenarios: dict[str, dict]) -> list[str]:
    return list(scenarios)
