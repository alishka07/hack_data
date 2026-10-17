# Hybrid 3.0 validation, 17 October 2026

The implemented service runs locally. These results establish software operation on the tested workloads, not release-quality violation detection.

## Verification

- Full Python suite: **720 passed** in 16.18 seconds. API tests ran with the Windows event-loop socket capability unavailable in the restricted sandbox. One existing Starlette/httpx deprecation warning remains.
- Browser regression: passed in headless local Chrome using synthetic intercepted requests. New checks cover 24/48/50-item pagination, deferred authenticated JPEG loading, exception evidence, source-time seeking, dense coverage metrics, safe text rendering and mobile layout.
- Python compilation and JavaScript syntax checks passed.
- Annotation workflow: seven local CLI stages passed on generated synthetic media. No real reviewed annotations were created. See [the workflow report](annotation-workflow-smoke.json).
- YOLOX versus genuine ByteTrack replay completed in an isolated evaluation environment; no GPU or media reads were used for that replay. See [tracking comparison](tracking-comparison.json). Production retains adjacent-frame geometric association.

## Final local smoke measurements

Both accepted jobs completed within their 300-second deadline and passed authentication, invalid-input rejection, busy admission, exact jury schema, source-PTS and normalized-box checks. Each benchmark checked three declared authenticated JPEGs; all emitted jury JPEGs would be checked, but neither run emitted a confirmed finding.

| Metric | Short clip | Longest clip |
| --- | ---: | ---: |
| Source duration | 11.633 s | 142.533 s |
| Upload to completed result | 60.594 s | 128.266 s |
| Decode/cache measurement | 1.726 s | 15.898 s |
| CPU perception | 1.270 s | 13.745 s |
| Local VLM calls | 53.349 s | 82.851 s |
| Evidence/output measurement | 0.040 s | 0.759 s |
| Completed/planned screen windows | 3/3 | 20/20 |
| Person/motion source samples | 59 | 714 |
| Final source frame included | yes | yes |
| Unconfirmed review items | 5 | 1 |
| Emitted jury findings | 0 | 0 |
| Highest sampled whole-GPU use | 7,589 MiB | 7,688 MiB |

GPU total memory is 8,151 MiB. Values are approximately one-second whole-device samples, not true instantaneous or model-only peaks. The phase measurements cover their named operations and are not a complete partition of total runtime; cache refinements, bookkeeping and other host work account for the remainder. All four recorded stage targets were met on these runs. This does not guarantee the same timing on a candidate flood or a different video.

Source reports: [short](hybrid-v3-shortest.json), [longest](hybrid-v3-longest.json). During development, an earlier longest run failed on a semantically unusable optional scene estimate. The final implementation discards such geometry to explicit unknown, records limited coverage and keeps its raw proposal in the private audit. Foreign source references and malformed response structure still fail. The final rerun above passed; development attempts must not be mistaken for an independent reliability study.

## Detection quality remains unfinished

The exact-version evaluator selects the latest completed 3.0 result per source hash, covering only **2 of 40 unique videos**. The short positive reference was missed by the jury verdict; the longest negative reference was correct. Thus this sample has TP=0, FP=0, TN=1 and FN=1. Review items do not change that confusion matrix.

The proposal stage covered the one evaluated weak positive timestamp within three seconds, with an absolute timing difference of 0.6 seconds. That is one point on previously inspected footage, not demonstrated event recall, a correct category or a justified legal finding. No real emitted positive evidence was available to validate human evidence relevance.

[The aggregate evaluation](evaluation-v3-0.json) preserves completed-group metrics, weak-point localization and unevaluated coverage. Its database-wide status buckets include historical jobs and must not be treated as attempt-level reliability for this final build. The per-run benchmark reports retain their own final processing outcomes.

The R3D-18 research model remains disabled after its failed gate. Real event/category/interval annotations, tracking ground truth where needed, independently reviewed evidence and a separate camera-angle test remain necessary before claiming useful detection quality or generalization.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts/run_local.py --port 8011
.\.venv\Scripts\python.exe scripts/check_hybrid_local.py --url http://127.0.0.1:8011 --sample shortest --output docs/hybrid-v3-shortest.json
.\.venv\Scripts\python.exe scripts/check_hybrid_local.py --url http://127.0.0.1:8011 --sample longest --output docs/hybrid-v3-longest.json
.\.venv\Scripts\python.exe scripts/evaluate_local.py --pipeline-version 3.0 --output docs/evaluation-v3-0.json
.\.venv\Scripts\python.exe -m pytest -q --basetemp=tmp/pytest-hybrid-fresh
```

Run only one launcher/worker. For annotation commands and reviewed import requirements, see [annotations.md](annotations.md). For setup, source provenance and remaining scope limits, see [hybrid architecture](hybrid-architecture.md).
