# Local reviewed event annotations

`worker/annotations.py` defines frozen schema `forte-reviewed-events/1`.
`scripts/review_annotations.py` creates drafts, validates them against actual
local video bytes and source presentation timestamps, and imports explicit human
reviews. These are evaluation tools. Production inference must never load these
labels, source hashes, categories, or reference intervals to decide predictions.

The repository's dataset audit contains 42 weak binary rows, 40 unique video
hashes, 21 positive groups, and 19 negative groups. Those weak labels and point
anchors are not reviewed event boundaries. This implementation creates no human
annotations or training positives automatically.

## Create and review a draft

Run from the repository root with the existing FFmpeg/ffprobe configuration:

```powershell
.\.venv\Scripts\python.exe scripts/review_annotations.py init --video C:\local\clip.mov --output data\annotation-drafts\clip.json
```

The command writes an empty draft and a private `clip.pts.json` sidecar containing
the exact source frame indices and normalized PTS. It preserves the absolute
first source PTS and a digest of the complete PTS sequence in the draft's source
identity. It does not copy media or infer a label. Keep both files local.

Inspect the original video locally and edit the draft JSON. The fields are:

| Field | Required meaning |
| --- | --- |
| `source` | Immutable source byte hash, geometry, duration, frame count, first PTS and PTS digest |
| `groups` | Explicit `camera_id`, `session_id`, and optional `related_group_id` |
| `review.status` | `draft` or `reviewed` |
| `review.reviewer`, `reviewed_at` | Named reviewer and ISO date/time with timezone |
| `review.human_review_attested` | Explicit assertion that a human reviewed this annotation |
| `review.coverage_intervals` | Reviewed intervals with `start_sec` and `end_sec` |
| `review.coverage` | `partial` or `complete`; complete coverage cannot contain gaps |
| `review.video_verdict` | `violation`, `no_violation`, or `unknown`, based on reviewed observable evidence |
| `review.unknowns` | Visibility or interpretation limitations; initially empty rather than invented |
| `events` | Reviewed event candidates, including supported, contradicted and unknown candidates |

Each event requires `event_id`, `type`, `start_sec`, `end_sec`, `status`,
`visible_facts`, `prerequisites`, `exceptions`, `uncertainty`, and `evidence_refs`.
Types are identifiers from the local catalog; `other_observable_event` and
`unknown` support uncategorized candidates. `unknown` cannot be a supported event.
An observed action label does not establish legal liability or rule applicability.

Each fact contains a unique `name`, `state` (`supported`, `contradicted`, or
`unknown`), a `description`, and `frame_refs`. A frame reference contains
`source_index` and the **exact** `timestamp_sec` from the PTS sidecar. Supported
and contradicted facts require source frames; unknown facts may have none.
Before/after context facts may cite frames outside the action interval.

`evidence_refs` use the same source reference plus `relevance` (`relevant`,
`irrelevant`, or `unknown`). Event evidence must be inside the event interval.
Only a human should assign relevance, and `unknown` is appropriate without that
review. A supported event requires source evidence and at least one supported
visible fact. Record visibility and exception uncertainty explicitly.

Set `status` to `reviewed`, name the reviewer, set a timezone-aware review time,
and set `human_review_attested` to true only after human inspection. Reviewed
imports require known camera/session grouping. Use the same session identifier
across related clips, including clips from different cameras in that session.
If metadata is unavailable, retain a draft until grouping is reviewed; assigning
distinct arbitrary identifiers does not establish independent footage.

A negative video label requires complete coverage, no supported events, no
unknown events, and no recorded unresolved review unknowns. Otherwise retain
`unknown`. Unannotated intervals and unlisted categories remain unknown; they
are never implicit training negatives.

## Validate and import

```powershell
.\.venv\Scripts\python.exe scripts/review_annotations.py validate --video C:\local\clip.mov --annotation data\annotation-drafts\clip.json
.\.venv\Scripts\python.exe scripts/review_annotations.py import --video C:\local\clip.mov --annotation data\annotation-drafts\clip.json --attest-reviewed "Reviewer name"
```

Import rechecks actual bytes, geometry, complete source PTS identity, every cited
frame, event bounds and human review metadata. `--attest-reviewed` must exactly
match the named reviewer. This assertion records review provenance; software
cannot prove that the human interpretation is correct. Formula-like strings,
duplicate JSON keys, nonfinite numbers, URLs and UNC source paths are rejected.
No spreadsheet formulas are evaluated.

The canonical destination is `data/annotations/<source-sha256>.json`. Publication
uses a same-directory temporary file, flush, and atomic exclusive publication
(rename on Windows; hard link on POSIX).
An identical reimport is idempotent; a different existing file is never replaced.
Review a conflicting revision manually and explicitly archive the old canonical
file before importing a replacement. Duplicate source copies share one canonical
annotation. The commands print aggregate counts only; rejection output never
prints private annotation text or source paths.

## Evaluate reviewed intervals

Use a private JSON source manifest with one exact path per hash:

```json
{"sources": [{"video_sha256": "<64 lowercase hexadecimal characters>", "video_path": "C:\\local\\clip.mov"}]}
```

Prepare local predictions independently of those labels:

```json
{"predictions": [{"video_sha256": "<same source hash>", "status": "done", "verdict": "violation", "review_required": true, "proposals": [{"start_sec": 1.0, "end_sec": 3.0}], "violations": [{"type": "<catalog identifier>", "timecode_sec": 2.0, "evidence_frames": [{"timestamp_sec": 2.0}]}], "latency_sec": 12.0}]}
```

Only include `peak_gpu_memory_mib` if peak memory was actually measured; a sample
after processing is not a peak. The prediction format is an evaluation adapter,
not a change to the jury API. Do not pass either manifest into inference.

```powershell
.\.venv\Scripts\python.exe scripts/review_annotations.py evaluate --sources data\annotation-sources.json --predictions data\annotation-predictions.json --output data\annotation-metrics.json
```

Evaluation revalidates imported annotations against their local source videos.
`proposal_metrics` reports explicit numerators, denominators and missing results:

- Proposal recall uses reviewed supported event intervals and temporal IoU
  (default threshold 0.1), independent of proposed category.
- Event localization uses same-category timestamps within a reviewed interval
  plus a declared tolerance (default 3 seconds). A finding matches at most one
  event. Its precision denominator includes only findings near a supported or
  contradicted same-category candidate interval; unlisted categories and other
  intervals remain unassessed. This temporal measure does not establish legal
  correctness.
- Video precision, recall and specificity include only completed predictions
  with explicit human video labels. Failed or pending jobs never become clean
  predictions. Their event references remain misses in event/proposal recall.
- Missing predictions are counted separately and excluded from matched-source
  rates. Unknown human video labels stay out of binary confusion.
- Temporal evidence relevance means a timestamp lies in a same-category reviewed
  event interval. Human evidence relevance uses only exact timestamps with an
  explicit human relevance label. Without those labels it remains `null`.
- Review and failure rates use all matched source records; runtime and memory
  summaries state how many sources have actual measurements. An absent
  denominator yields `null`, never an invented perfect score.

The metrics helper accepts already validated annotations. The CLI performs
source revalidation before calling it. Review every reported result locally;
these tools do not claim annotation accuracy or detector quality.

The [synthetic workflow smoke](annotation-workflow-smoke.json) exercises actual
FFmpeg generation/probing and CLI draft creation, validation, import, idempotent
reimport, and evaluation. It uses a two-second generated color clip with ten
source frames and simulated reviewer metadata only under
`tmp/annotation-workflow-smoke/`. It creates no human reviews, modifies no
production annotations, and leaves the reference video verdict `unknown`.
Binary confusion has zero eligible samples and quality metrics remain `null`.
The 35 synthetic annotation unit tests also cover exact frame binding, source
identity, rejected drafts, exclusive import, grouping and metric denominators.

The generated local fixture can be revalidated and reimported with:

```powershell
.\.venv\Scripts\python.exe scripts/review_annotations.py validate --video tmp\annotation-workflow-smoke\synthetic.mov --annotation tmp\annotation-workflow-smoke\reviewed-fixture.json
.\.venv\Scripts\python.exe scripts/review_annotations.py import --video tmp\annotation-workflow-smoke\synthetic.mov --annotation tmp\annotation-workflow-smoke\reviewed-fixture.json --attest-reviewed SYNTHETIC_WORKFLOW_FIXTURE --directory tmp\annotation-workflow-smoke\imports
.\.venv\Scripts\python.exe -m pytest tests/test_annotations.py -q --basetemp=tmp/pytest-annotations-fresh
```

## Training partitions

`split_group_ids` computes transitive components across duplicate source hashes,
shared sessions and explicit related groups. Partition complete components,
then fit preprocessing, feature scaling and thresholds only in training folds.
Use camera holdouts separately when evaluating a new view. The existing
`worker.training_data.deterministic_split` groups byte duplicates only: it must
not be treated as session-independent merely because these new annotations
exist. No automatic migration changes historical training results.

Annotate individual positive event intervals. Do not mark every window of a
positive video positive; unannotated windows remain unknown. Do not tune to
final evaluation footage. Existing corpus inspection means historical internal
splits remain exploratory rather than untouched external tests.
