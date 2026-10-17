# Invalid-output fixes, pipeline 5.1

Recent local jobs exposed two concrete defects: schema-valid repeated scenario IDs caused `model_schema_invalid`, and verification responses could claim `supported` while their checks said `no`/`not_visible`, visibility was unusable or evidence was empty. Negative-only observations also appeared as unconfirmed categories in the UI.

The fixes are:

- Allow the same scenario at different source intervals and merge duplicate category/interval proposals. Canonicalize misordered proposal bounds using only the supplied valid source references, preserving the cited anchor. Invalid IDs, types and out-of-range references remain rejected.
- Suppress explicit negative-only observations as proposals. Mixed positive evidence such as visible cash counting without a client remains eligible; a missing procedure is not automatically treated as an empty observation.
- Normalize every action/context verification conservatively. Explicit contrary checks or applicable exceptions reject support. Unknown/hidden checks, limited visibility, invalid confidence, missing citations and placeholder descriptions cannot retain a supported decision. Unknown checks cannot retain affirmative source citations.
- Keep rejected and evidence-free hypotheses in diagnostic records, outside the active review queue. Unresolved verification is recorded as a coverage limitation. Negative rule applications cannot reappear as catalog alerts.
- Derive displayed counts from active host cases. Rejected or explicitly dismissed cases cannot inflate counts, stale summaries cannot override an explicit case list, and unsupported model prose is hidden. Genuine visibility and unresolved-verification gaps remain visible.

The full regression suite passed **945 tests in 27.54 seconds**. Headless Chrome checks passed, including rejected/stale counts, empty active-case lists, genuine visibility gaps and unresolved-verification coverage. These checks validate the fixes, not overall detector accuracy.

[The authenticated upload replay](invalid-output-fix-replay.json) records the separate result of retrying a source whose earlier job failed during scenario parsing. The earlier job and source files remain unchanged.

The replay completed successfully on version **5.1** in **150.689 seconds**, with no schema error. It corrected two contradictory model decisions, dismissed one unsupported hypothesis and retained two evidence-linked review items. No automatic violation was confirmed. The local worker returned to ready afterward. This validates the previously failing operational path; it is not an independent accuracy benchmark.
