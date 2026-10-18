# Scenario-first video review, version 5.5

Version 5.5 adds [departure with a computer screen left on](workplace-screen.md), under the user's workplace policy. It locates computers through OWLv2 and source-linked VLM detections, proposes a per-video computer region for detailed views, and separately verifies the departure and screen state. A finding needs a before/exit/after sequence and a post-departure frame showing the same computer screen visibly on at an unattended workstation. Primary evidence is pinned to that post-departure state and includes before/after JPEGs. The food-belongings policy remains active.

Version 5.4 replaces the eating-only rule with the user's [personal-food belongings policy](workplace-belongings.md). Clearly identifiable food on the operating cashier work table is sufficient for a workplace-policy violation; no eating or consumption sequence is required. Food identity, table location and work context must share a cited source frame. A single readable frame can establish this static condition. The older eating rule is disabled for new analyses; saved results are preserved.

Version 5.3 adds the [workplace eating policy](workplace-eating.md). Supported food consumption by the operator at the cashier work table can emit a workplace-policy violation with source timestamps and JPEG evidence. Two distinct source times, visible work context, and explicit alternative-action checks are required. The source is the user's policy; the three public-regulation gates retain their existing requirements. Older completed jobs retain their original results.

Version 5.2 adds [candidate consolidation and deadline-aware verification](deadline-fix.md). Each model request preserves a finishing reserve. Checks that cannot be completed remain explicitly unverified, and the UI displays Review incomplete. Full timeline screening remains required before a result can be assembled.

Version 5.1 adds [semantic output validation and duplicate-proposal fixes](invalid-output-fix.md). Rejected and evidence-free model guesses remain diagnostic records; they do not inflate the active review count. The version 5.0 measurements below are historical and remain separate from the fixed-version replay.

Version 5.0 closes missing detection paths for the category names in `open/open_labels.simplified.xlsx`. The fixed taxonomy lives in `rules/scenarios.json`; inference never opens the workbook, uses example filenames or looks up reference verdicts. Its nine scenarios include all seven hinted types, plus workstation departure and payout-count review.

## Processing

1. Keep source-timestamp ingestion, anonymous ByteTrack, RTMPose, sparse OWLv2, per-video zones and temporal proposals from [version 4.0](architecture-v4.md).
2. Add a bounded scenario checklist to each existing timeline-screening request. The model can return up to six distinct hypotheses with supplied frame indices, observed/uncertain states and uncalibrated confidence. This adds no separate screening call.
3. Independently seed scenario proposals from positive source-linked key/stamp, personal-object, storage-transfer and cash-processing observations. Deduplicate overlapping episodes before verification.
4. Verify a manual scenario against its explicit visual predicates using up to eight detailed onset/action/ending views. A supported visible cue no longer disappears as an unsupported category. Hidden/ambiguous predicates remain `not_visible`/unknown.
5. Retain a source-linked review item and up to three original-dimension JPEGs. The UI distinguishes verified visible scenarios from unresolved proposals. Seeking remains tied to source timestamps. `details.scenario_coverage` records screening, proposals and verified visual episode counts by family.
6. Apply the existing independent transaction, applicability, exception and legal gates to eligible automatic rules. Visible objects, unspecified staff roles, internal policies or unobserved procedures do not become automatic legal violations.

The seven hinted families are prior cash exposure, prior cash storage, cash intake without a client, back-office processing control, vault/storage segregation, keys/stamps handling and personal belongings. [The coverage report](label-coverage-v5.md) details their observable predicates. Vault storage was missing from the catalog and is now an explicit unmapped review concern. Storage/exposure aliases record related duties; their review outputs are not independent accuracy examples.

## Speed and scope

No new training, model download or per-family full-video pass is required. The broad checklist shares existing window calls; only proposed manual scenarios receive one additional focused visual check. Person/object cadence and the 300-second job deadline remain unchanged. More proposals can increase verification work, so the previous timing measurements do not establish version 5.0 latency.

The public jury JSON contract remains unchanged. Source-linked visual scenarios live in internal/UI details and review items. Only the three existing legal event rules can emit automatic findings. Counting omission remains review-only. Internal policy, ownership, staff authorization and hidden procedural completion require appropriate evidence.

## Validation

The complete regression suite passed **845 tests in 24.69 seconds**. Headless Chrome's existing review regression passed. New integration cases verify keys/stamps, belongings, back-office processing, vault storage, prior storage and payout review; they check original JPEG dimensions, declared evidence, source references, unknown handling and absence of unsupported legal findings. Synthetic tests establish software behavior, not detection recall.

[The local-model request check](scenario-v5-runtime-check.json) exercises the expanded strict schema on eight actual source frames. It returned a valid response with zero scenario hypotheses and unusable visibility; this is a compatibility result, not a successful positive detection. [The complete short-video check](architecture-v5-shortest.json) completed in **64.229 seconds** on the resident 4B server, produced six review items and validated 62 declared JPEGs. It emitted zero automatic findings and zero host-supported manual scenarios. The known positive reference therefore remains unconfirmed. This check exercised the first scenario prompt revision; a subsequent bounded prose-only prompt improvement passed its focused regression checks, without another complete video accuracy run.

The workbook has eleven positive rows without category hints. Its hinted categories also occur on negative examples. Covering all seven names therefore does not establish detection of every positive video. No perfect-recall claim, automatic label lookup or source-specific prediction override is supplied.

## Run

Use the existing local launcher and installed models:

```powershell
.\.venv\Scripts\python.exe scripts/run_local.py
```

If a trusted local model server is already running on port 18080, set `FORTE_ATTACH_MODEL=1` before launching; attached servers are never stopped by the app. Open the local app at the launcher's printed URL. Existing [setup instructions](architecture-v4.md) and [verifier configuration](vlm-runtime.md) remain applicable.
