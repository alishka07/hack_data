# Deadline and model-request fix, pipeline 5.2

The failing upload was 25.933 seconds long. It generated 12 screening windows and 60 raw scenario rows; 42 source-bound scenario proposals survived parsing. Full timeline screening finished approximately 144.8 seconds after acceptance. Repeated candidate verification exhausted the remaining budget. The final request timed out at approximately 295 seconds because the runtime reserved five seconds, then classified the timeout as a generic model/structured-output failure. The source and its failed job are retained.

Version 5.2 changes the work scheduling and completion boundary:

- Consolidate substantially overlapping candidates within the same rule/episode, preserving source spans and distinct separated events. Give each rule type a first turn before repeated candidates of another type.
- Check admission before every verification/refinement/context request, using recent request latency and a 25-second finishing reserve. The runtime also receives the earlier verification deadline, so a slow request cannot consume the final output reserve.
- Avoid optional repeated checks when the relevant view is already unusable. Hidden predicates remain unknown.
- Retain verified findings and source evidence if a later candidate check cannot finish. Deferred candidates stay in a separate diagnostic list with a reason and `verified=false`; they cannot become a finding or a fabricated active case. Full timeline screening is required before this result can be assembled.
- Set `processing_status=review_incomplete`, `coverage.verification_complete=false` and `unverified_candidate_count` when needed. The UI displays Review incomplete and Needs review. The jury JSON remains unchanged; API consumers needing completeness must inspect the authenticated UI-job detail fields.
- Distinguish `model_timeout`, `deadline_exceeded`, `model_http_error`, `model_transport_error` and `model_output_invalid`. JSON/schema errors and HTTP 502/503 get at most one retry when a full request allowance remains; timeouts and truncated replies are not blindly retried.
- Record safe request counts, durations and error codes in runtime metadata, and checkpoint verification attempts. Diagnostic metadata contains no prompts, images or response bodies.

The full regression suite passed **981 tests in 27.08 seconds**. Chrome checks passed for incomplete review, retained confirmed findings, actual unfinished-check counts and the existing invalid-output protections. New synthetic cases cover budget refusal, a late timeout/invalid response/connection failure, preserved earlier findings and unresolved hidden views. These tests establish software behavior rather than detector accuracy.

[The exact-upload replay](deadline-fix-replay.json) records actual execution, completion scope, source-image validation and request diagnostics on the resident local 4B verifier. Its measurements are separate from synthetic tests and do not establish all-case recall.

The replay returned results in **268.270 seconds**, with all **12/12** timeline windows screened. Consolidation reduced 42 retained proposals to 32 candidate episodes. Sixteen candidates remained unverified and were explicitly reported as **Review incomplete**; seven evidence-linked review items were retained and all 141 declared JPEGs decoded successfully. The runtime recorded 37 requests, zero retries and zero request failures, and the worker returned to ready. Complete verification of this proposal-heavy clip inside five minutes remains unachieved; the failure-handling fix preserves usable results and makes that limit visible. VLM work took 218.851 seconds, exceeding its 150-second engineering target.
