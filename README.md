# ForteBank local video review

For a fresh laptop, follow the [portable setup guide](docs/local-setup.md). The setup downloads public model weights separately; private videos, API keys and saved jobs are not part of this repository.

A local prototype for reviewing silent cash-desk videos. Pipeline **5.3** screens all seven category names in `open_labels.simplified.xlsx`, within ten curated scenarios, including the [user-provided workplace eating policy](docs/workplace-eating.md), and validates model decisions against their checks, visibility and citations. It consolidates repeated candidates and reserves time to save results before the deadline. Unfinished checks are explicitly marked for review. It combines YOLOX + genuine ByteTrack, local RTMPose, sparse OWLv2 objects, per-video scene proposals, temporal events, bounded Qwen verification and deterministic rule/exception gates. FastAPI serves the UI and jury API; one persistent worker uses the local verifier.

The current implementation is [architecture 5.3](docs/architecture-v5.md), with [deadline handling](docs/deadline-fix.md), [invalid-output fixes](docs/invalid-output-fix.md), [label coverage](docs/label-coverage-v5.md) and the [base model setup](docs/architecture-v4.md). The labels leave eleven positive rows without category hints; covering the named types does not establish detection of every positive case. [The data audit and fine-tuning analysis](docs/finetuning-analysis.md) covers all 402 videos in `open` and `clips`. The 360 augmented files remain grouped and quarantined until source/label-preservation provenance is reviewed. The existing 4B GGUF remains the default verifier. Compatible [7B four-bit weights and projector](docs/vlm-runtime.md) are installed, hash verified and tested with four synthetic images on CPU; 7B GPU/video performance remains unvalidated.

[Historical 3.0 validation](docs/hybrid-validation.md) completed short/long local runs in 60.594/128.266 seconds but missed the positive reference. These measurements do not validate 4.0 detection accuracy or timing. Current checks and remaining limits are recorded in the 4.0 guide; no detector, pose or VLM fine-tuning improvement is claimed.

The [local annotation workflow](docs/annotations.md) validates source hashes and frame PTS, imports explicitly reviewed records and evaluates reviewed intervals. [CPU perception setup and provenance](docs/perception.md) documents the installed detector, tracker, pose and object models. The [R3D-18 comparison](docs/event-model-evaluation.json) failed its release gate (39.35% balanced accuracy), so that optional learned temporal scorer remains disabled. The sections below describing pipelines 1.x–3.0 are historical.

For a fresh install, download the public person model once before offline use:

```powershell
.\.venv\Scripts\python.exe scripts/setup_perception.py
.\.venv\Scripts\python.exe scripts/run_local.py
```

Benchmark the running service with `scripts/check_hybrid_local.py --sample shortest --output docs/architecture-v4-api-shortest.json`, then repeat with `--sample longest --output docs/architecture-v4-api-longest.json`. These tests upload generic-named video bytes and publish only aggregate operation metrics.

The implementation does not establish compliance or legal liability. `no_violation` means no supported finding in the verified scope. Failures before full timeline screening return `failed`. Later candidate-verification limits preserve supported results and explicitly mark the UI review incomplete; inspect `details.coverage.verification_complete` before treating a result as fully checked. Detection accuracy, unseen-category recall and camera generalization require independently reviewed evaluation.

The original labels have been restored. The current workbook SHA-256 is `4faf595a2679c2f8b2b41be041de59bb3c11cd6e77d57226f1cbcc14b2cc3e75`: 42 rows contain 22 positives and 20 negatives, representing 40 unique video hashes (21 positive, 19 negative). Column C's explicit verdict is authoritative; column B supplies weak free-text time anchors, with 22 rows parsed. A blank timestamp does not override an explicit verdict. One duplicate hash group has disagreeing time anchors and is excluded from localization scoring. These anchors do not establish event categories or reviewed event boundaries. See [the aggregate dataset audit](docs/dataset-audit.json).

Earlier results against the incomplete all-blank reference are superseded and retained in [the archived report](docs/evaluation-allblank-superseded.json). Labels and annotation text never enter inference.

A separate supervised classifier was trained from `open/open_labels.simplified.xlsx`, keeping duplicate videos together. It scored 50% on eight held-out videos and falsely flagged all four clean videos, so it remains experimental and is not used by the website. See [training commands and limitations](docs/training.md) and [the measured results](docs/trained-case-evaluation.json).

## Start on this laptop

Use Python 3.12 or newer, the NVIDIA driver, FFmpeg/ffprobe, and the already downloaded local model/runtime. Install dependencies once with internet access:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe scripts/run_local.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The launcher prints the API-key file path. Open `data/api-key.txt` locally, paste the key into the page, and upload a `.mov` file. The browser retains the key in memory. `--show-key` explicitly displays it in the user's terminal if preferred. Stop both API and worker with Ctrl+C.

The service processes one video at a time. While it is busy, the page disables new analysis and shows the running review. After a refresh, entering the key resumes an active review automatically; if a file or another review is already selected, use **Open running analysis**. Pausing browser updates leaves the server analysis running. Authenticated clients can discover the current processing job at `GET /api/ui/active-job`; an idle worker returns `{"job_id":null}`.

The launcher uses the project `.venv` if present, starts children from the repository directory, and stops only its owned process groups. The worker starts/stops its own model server. Do not run a second worker against the same GPU/database. Startup can take time while model weights load; wait for the service's readiness indicator before submitting.

Default model artifacts are private and ignored by Git:

```text
tmp/planning/local_runtime/bin/llama-server.exe
tmp/planning/local_runtime/Qwen3VL-4B-Instruct-Q4_K_M.gguf
tmp/planning/local_runtime/mmproj-Qwen3VL-4B-Instruct-F16.gguf
```

If these paths or FFmpeg differ, set environment variables before starting:

```powershell
$env:FORTE_LLAMA_SERVER = 'C:\path\to\llama-server.exe'
$env:FORTE_MODEL_PATH = 'C:\models\Qwen3VL-4B-Instruct-Q4_K_M.gguf'
$env:FORTE_MMPROJ_PATH = 'C:\models\mmproj-Qwen3VL-4B-Instruct-F16.gguf'
$env:FORTE_FFMPEG = 'C:\path\to\ffmpeg.exe'
$env:FORTE_FFPROBE = 'C:\path\to\ffprobe.exe'
.\.venv\Scripts\python.exe scripts/run_local.py
```

`FORTE_DATA_DIR` selects local storage; default `data/` holds the SQLite database, API key, uploads, and evidence. `FORTE_API_KEY` overrides the persisted key when explicitly provided. `FORTE_HOST` and `FORTE_PORT` select the local bind address/port; this launcher permits loopback addresses only. Use `--port 8001` if 8000 is occupied. All frontend assets are served locally with no CDN. After installing dependencies and weights, processing needs no external network.

## Jury API

Both requests require `X-API-Key`. `POST /api/v1/analyze` accepts multipart field `file`; `GET /api/v1/jobs/{job_id}` polls the result. Example from a local PowerShell terminal (the key is read without printing it):

```powershell
$forteKey = (Get-Content -Raw -LiteralPath '.\data\api-key.txt').Trim()
$accepted = curl.exe --silent --show-error -X POST `
  -H "X-API-Key: $forteKey" `
  -F 'file=@C:\path\to\clip.mov;type=video/quicktime' `
  http://127.0.0.1:8000/api/v1/analyze | ConvertFrom-Json
curl.exe --silent --show-error `
  -H "X-API-Key: $forteKey" `
  "http://127.0.0.1:8000/api/v1/jobs/$($accepted.job_id)"
```

Accepted uploads receive HTTP 202 and a `job_id`. Public statuses are `processing`, `done`, or `failed`. A successful response has verdict `violation` or `no_violation`:

```json
{
  "job_id": "uuid",
  "status": "done",
  "verdict": "violation",
  "violations": [{
    "type": "active_intake_departure",
    "timecode_sec": 12.5,
    "description": "Possible violation: supported action and context; Resolution 231, paragraph 22; official source URL.",
    "evidence_frames": [{"timestamp_sec": 12.5, "image_base64": "<JPEG base64>"}]
  }]
}
```

`no_violation` uses an empty `violations` array. Failed jobs contain no completed verdict. Authentication failures are 401; unknown jobs 404; oversized input 413; overload 429 with `Retry-After`. One admitted job at a time protects the 8 GB GPU and limits queue delay. The deadline is 300 seconds from acceptance. After full timeline screening, a result can retain verified findings with `details.processing_status=review_incomplete` and an explicit unverified-candidate count in the authenticated UI-job endpoint. The UI displays Needs review. The public jury JSON is unchanged; consumers needing completeness must inspect these detail fields.

## Pipeline 2.4, enabled scope and evidence

The worker records structured action facts, source-frame indices/timestamps, transaction cues and catalog estimates in a local ledger. Eleven visible checks receive source-linked scores in the existing window model calls before legal confirmation; scoring adds no model calls. Cached frames support dense action checks before the transaction-context gate. When action checks remain unknown, an expanded onset-to-ending view is inspected before rejection. Uncertain legal context can receive a separate targeted before/action/after refinement. Each yes predicate must cite supplied frames; unshown ledger notes cannot establish it, and an earlier explicit rejection is preserved. Episode-aware deduplication merges repeated observations of the same event while preserving distinct episodes.

Screening retains eight ordered samples per window. Three anchors, at request positions 0, 3 and 7, use original source resolution capped at 960 pixels on the long side; the other five retain height 448. This preserves timeline coverage while giving the model more detail at the beginning, middle and end. The model context is 8,192 tokens, with a bounded object-localization response in the existing screening calls. The changed prompts and context require fresh latency and GPU-memory measurements; earlier results do not validate this configuration.

Each localized person or physical item has a labeled, normalized box tied to a supplied source frame and timestamp. The UI provides annotated item cards and boxes on the matching sampled video frame. These are model-estimated locations, not continuous tracks; an object box does not establish a violation, cash ownership or transaction continuity. At most twelve object annotations are returned per screening window, with repeated annotations of the same object on the same source frame deduplicated. Source JPEGs use the authenticated evidence route.

The case decider shows supported violation evidence, review required, or no supported violation. Unresolved action/context checks, pending source-linked assessments and visual gaps lead to review required rather than a clean-result headline. Each case lists its action and applicability checks, explicit contrary evidence, unknowns, reasons and available source timestamps. Only findings that pass the existing evidence and rule gate count as violations in the jury API; its binary response contract remains unchanged. Detection boxes and scores alone cannot create a confirmed finding.

The v2.4 local smoke run processed an 11.633-second clip in 40.375 seconds, returned eight localized items on one source frame and four cases marked review required, and emitted no confirmed findings. All eight boxes passed coordinate checks; the declared source JPEG decoded and required authentication. Six action checks were exposed across the cases. See [the v2.4 smoke report](docs/local-confidence-v2-4-smoke.json). A post-run GPU sample was 7,267 of 8,151 MiB, not a peak-memory measurement. This checks operation on one clip; box precision, detection recall and longest-clip timing still require evaluation for 2.4.

Three categories can produce automatic findings when their visible actions, context and exceptions are supported. The payout-count omission category is review-only:

| Type | Emission policy | Required context | Public rule |
| --- | --- | --- | --- |
| `active_intake_departure` | `auto_emit` | Established unfinished intake and actual workstation departure | Resolution 231, paragraph 22 |
| `cash_intake_without_client_presence` | `auto_emit` | Identified client intake, visibly absent client, acceptance/counting | Resolution 231, paragraph 20 |
| `previous_transaction_cash_exposed` | `auto_emit` | Two distinct transactions, continuous earlier-cash identity, cash outside storage | Resolution 231, paragraph 20 |
| `loose_cash_issued_without_observed_count` | Review-only | Complete visible payout and counting opportunity; earlier/offscreen/machine counting and packaging exceptions remain important alternatives | Resolution 231, paragraphs 31(4), 32 |

Every required predicate must be explicitly supported. Unknown or occluded prerequisites do not satisfy the rule. Citations are assembled from `rules/registry.json`. Cup/beverage, unseen physical ID, ordinary transaction without a supervisor, and generic open-door categories remain disabled. See `rules/sources/` for operative limits and version provenance.

Unsupported visible actions, uncertain prerequisites, and review-only omissions can enter a separate **UNCONFIRMED review queue**. The UI shows its timestamps, reasons and local observations alongside sampling coverage and analysis limitations. Queue items are not jury violations, do not increase the finding count, and do not change the public verdict. Coverage reports describe sampled windows and source-frame metadata; they do not establish uninterrupted visibility or compliance. The authenticated UI details retain these fields without extending the jury API.

No face recognition, identity inference, audio analysis, or remote model calls are used. Uploaded filenames and the labels workbook do not enter model inference. Keep the dataset, uploads, model observations, and evidence local. `open/`, `tmp/`, `data/`, and `models/` are ignored by Git.

## Review catalog and continuous scores

[The 34-item catalog](docs/violation-catalog.md) records applicability, evidence, exceptions and review routes; [the JSON catalog](rules/catalog.json) supplies the UI. Its added vault-storage concern has an explicitly unmapped legal source. Seven event checks initialize at **0%** and rise with model confidence in their occurrence: `max(previous score, 100 × confidence)`. Twenty-seven required-procedure checks initialize at **100%** and fall with confidence that the correct protocol was completed: `min(previous score, 100 × (1 − completion confidence))`. These uncalibrated review estimates never create a finding by themselves.

After entering the API key, the UI shows the catalog through authenticated `GET /api/ui/catalog`. New analyses retain all 34 states separately from the exact jury response. Version 5.0 screens nine explicit visual scenarios during existing timeline calls and verifies proposed manual scenarios using source-linked predicates. Verified physical cues appear as visible scenarios requiring policy review. Manual catalog scores do not become legal findings from that visual support. Correct procedure completion still requires its actual protocol, context and exceptions. Older saved jobs remain readable.

The Results table includes a Probability column with each source-linked model estimate. Its scope states whether it covers the visible action, all rule checks, event occurrence or correct procedure completion. These estimates are uncalibrated; absent, invalid, unobserved and manual scores display **Not assessed**. Candidate estimates stay tied to their own episode and timestamp. Existing completed reviews can display their retained verifier scores without reanalysis, while their stored verdicts and decisions remain unchanged.

Observed start cues in the host ledger create fresh transaction assessments. Without established boundaries, each window stays isolated. Within an episode, stronger evidence moves the estimate monotonically and duplicate/overlapping frames never add confidence. The summary retains the highest remaining estimate across relevant assessed episodes, preserving pending procedures even if another transaction was completed correctly. Scores are never added or averaged, and mirrored views of the same duty are linked to prevent double counting.

The summary covers explicitly assessed episodes. A known transaction without a returned assessment for that check is reported as incomplete coverage; a low score in another transaction does not certify that missing one. Window-only observations also show that their transaction boundary is unconfirmed. An unknown, untriggered window with zero support now records a coverage gap instead of creating a fresh 100% procedure episode that masks supported estimates from other windows.

This working catalog covers 4 automatic checks (3 public-regulation rules and the user-provided eating policy), 8 review-only checks and 22 manual checks. The inference pipeline is **5.3** and scoring remains **2.1**. The three public-regulation gates retain their existing conditions. Earlier reports do not establish version 5.3 accuracy or timing.

## Verify and operate

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app worker scripts
```

To prepare the deterministic exploratory split and exercise the running service with its local sample selection:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_local.py --prepare-validation
.\.venv\Scripts\python.exe scripts/benchmark_local.py --selection data/labels/validation-selection.json --output docs/local-integration-v2-4.json
.\.venv\Scripts\python.exe scripts/evaluate_local.py --pipeline-version 2.4 --output docs/evaluation-v2-4.json
```

The seed-17 split keeps duplicate hashes together: 32 development groups and eight validation groups, balanced four positive/four negative. The conflicting-timestamp duplicate stays in development. This corpus was previously inspected, so the partition is **exploratory validation, not an untouched test set**. Private manifests stay under `data/labels/`; only video bytes reach inference. The benchmark's selection contains a balanced subset of the validation groups, rather than claiming that all eight have been processed.

The benchmark checks authentication, invalid-input rejection, busy admission, exact response fields, screening completion, source timestamps, and JPEG decoding for any emitted evidence. It prints aggregate counts and timings only; full model observations stay in `data/`. The evaluator matches videos by byte hash and uses the latest completed result **with the exact requested pipeline version** per group. Keep the version filter to avoid mixing v1 and v2 jobs. The test suite also exercises multiple findings and evidence generation with synthetic media. These technical checks do not establish detection accuracy.

Under the restored labels, [the v1 exploratory baseline](docs/evaluation-v1-baseline.json) contains two false negatives and two true negatives across four matched groups. Earlier v1 runtime reports, including `docs/local-integration-optimized.json`, remain historical measurements. The completed v2.1 check covers five unique clips: three negatives were correct and two positives were missed (TP=0, FP=0, TN=3, FN=2). All five jobs completed in 13.204–130.078 seconds; the longest 142.533-second clip took 130.078 seconds. None emitted confirmed findings. Proposal timing covered one of two weak reference anchors within three seconds; this does not establish event type or legal correctness. Detection quality remains unfinished despite passing technical checks. See `docs/local-integration-v2-1.json` and `docs/evaluation-v2-1.json`. Earlier v2.0 reports are preserved separately and are not mixed into these metrics.

The automated Python suite covers catalog scope, continuous confidence, source-bounded evidence, duplicate-frame handling and per-episode scoring. The synthetic v2 browser regression also covers initial/assessed values, unknown and N/A states, coverage rendering, separation of unconfirmed queue items from findings, timestamp seeking, safe text rendering, mobile layout and older-result compatibility. `tests/ui_review_v2.cjs` runs with Node and Playwright browser-test tooling; its header documents optional module/browser overrides. It intercepts synthetic requests and does not access private jobs or media. If the Windows sandbox blocks temporary-directory creation, use a fresh workspace path with `--basetemp=tmp/pytest-<unique-run>`; API tests also require the local Windows event-loop socket capability.

Revision 2.2 passed **174 Python tests** and the synthetic browser regression, including event68% and correct-procedure82%/remaining18% examples. Two real local smoke tests completed: an 11.633-second clip in10.187 seconds and the longest142.533-second clip in149.266 seconds. The model returned6 and108 unresolved confidence observations respectively, with no changed or fractional scores and no findings. These runs verify operation and timing on those two clips; they do not demonstrate useful confidence estimates, accuracy, or calibration. See [short-run metrics](docs/local-confidence-v2-2-smoke.json) and [longest-run metrics](docs/local-confidence-v2-2-longest.json). Reproduce locally with `scripts/check_confidence_local.py`, adding `--longest` for the longest sample; only counts/timings are printed.

Revision 2.3 fixes two causes of unchanged scores. The previous prompt told the model to return zero for uncertain evidence, and the collector rejected an `unresolved` response carrying nonzero confidence. The revised prompt permits intermediate support; source-linked uncertain nonzero estimates are retained as `partial_support`, with the model's original state preserved separately. The episode adapter also stops unknown, untriggered windows from adding artificial pending procedures. **195 Python tests and the synthetic browser regression passed** after these changes. Focused local question probes still returned zero estimates; a combined screening probe using three source-resolution frames returned three uncertain nonzero estimates that the previous collector would reject.

The completed v2.3 reruns changed scores on real clips: the 11.633-second sample finished in 33.313 seconds, and the 142.533-second longest sample in 206.438 seconds. Each produced 4 fractional score changes (3 event increases, 1 procedure decrease), compared with 0 changes in v2.2. Both emitted 0 confirmed findings. These measurements establish working confidence updates and timing for these two clips, not accuracy or calibration. GPU memory was sampled at 6,643 of 8,151 MiB after the longest run; this is not a peak-memory profile. Reports include job IDs and phase timings: [short rerun](docs/local-confidence-v2-3-smoke.json), [longest rerun](docs/local-confidence-v2-3-longest.json). Those runs used three source-resolution anchors per eight-frame screening request, capped at 960 pixels on the long side, and a 6,144-token context.

For a local manual check, start the launcher, wait until ready, upload one valid MOV, and inspect status, verification completeness, source timestamps and clause applicability. Also check an invalid upload and a deliberately stopped worker. Decoder errors, stopped workers and incomplete timeline screening must fail. A late verification failure must remain an explicit review gap and cannot authorize a finding or a clean UI state. Mocked checks do not measure real video accuracy. Record actual timings and review every finding before reporting benchmark claims.

When troubleshooting, check the terminal's API/worker messages and model log under the local data directory. Missing model/runtime files require setting the path variables above. Missing FFmpeg requires installing it locally or setting both executable paths. A GPU allocation error requires closing other GPU applications or selecting a smaller supported model and remeasuring; CPU fallback does not promise the deadline. After an interrupted job, restart the service so lease/deadline recovery can resolve stale processing records.

No automatic deletion policy is advertised. Stop the service before manually archiving old completed-job data; deleting active uploads/evidence breaks processing and saved results. Keep storage access limited to intended local users. Before exposing the service publicly, add HTTPS and validate routing, upload limits, authentication, privacy, and timing with the actual jury access path.

## Model/runtime provenance

The local default is official Qwen3-VL-4B-Instruct GGUF, revision `1cd86afb9a95c410a6038ab3b40d8b578c892266`, Q4_K_M language weights and F16 vision projector. The pinned model card declares Apache-2.0; the download directory retains the card and Qwen code-license notice. A standalone model-repository LICENSE request returned 404 during planning; retain that distinction rather than asserting a file exists.

| Artifact | SHA-256 |
| --- | --- |
| Language GGUF | `66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a` |
| F16 vision projector | `256f3a43bd4205ffef48d6b92715e1e70b5b0e9aef06522584967513a9985331` |

Sources: [pinned Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/blob/1cd86afb9a95c410a6038ab3b40d8b578c892266/README.md), [llama.cpp b11707](https://github.com/ggml-org/llama.cpp/releases/tag/b11707). The downloaded Windows CUDA runtime is llama.cpp b11707, MIT; retained notices include `llama-LICENSE.txt` and LLVM OpenMP notice. NVIDIA CUDA binaries retain their own [NVIDIA terms](https://docs.nvidia.com/cuda/eula/index.html). The planning FFmpeg 9.0.1 binary was built with GPL/version3 enabled, so retain its provenance and GPLv3 obligations if redistributing that build; subprocess use does not remove distribution obligations. [FFmpeg legal page](https://ffmpeg.org/legal.html).

Direct Python versions are pinned in `requirements.txt` and `pyproject.toml` using public PyPI metadata checked 17 October 2026. Pydantic 2.12.5 and pydantic-core 2.41.5 are deliberately retained because the latter has a verified CPython 3.12 Windows x64 wheel; the latest Pydantic's required core lacked that installable wheel during setup. OpenCV and NumPy Windows wheel availability was also checked. Installed transitive versions are pinned in `requirements.lock.txt`. The installed license files and bundled notices are recorded in `docs/licenses.md`; retain their component-specific conditions when distributing. Architecture and research rationale are in `docs/architecture-plan.md`.
