# Eating at the cashier workstation

Pipeline 5.3 implements the user's instruction that cashiers must not eat at the work table during work. Rule `cashier_eating_at_workstation` is an automatic event under the explicitly recorded [user-provided workplace policy](../rules/sources/workplace-eating.md). It is not attributed to a public regulation.

All timeline windows screen for identifiable eating. A proposal then receives separate action and context verification. A supported finding requires visible food consumption across at least two distinct source images and times, linked to the same visible workstation operator, with the relevant work context established. Repeated crops of one image cannot satisfy the temporal requirement. The source may show an operating cashier desk without an active client transaction; a break area or unresolved work context cannot establish applicability.

Food merely on the table, a cup, drinking, medicine, touching the face, and ambiguous hand-to-mouth gestures are insufficient. Missing, negative, uncertain, unreadable or uncited checks cannot emit a violation. Explicitly negative or action-free model descriptions cannot override those conditions.

The result uses `type: cashier_eating_at_workstation`, the source `timecode_sec`, a workplace-policy description, and one to three evidence JPEGs. Its primary event time comes from a cited food-consumption frame. UI findings identify the policy basis and link to the local `/policies/workplace-eating` source. Existing results are not retrospectively relabelled; reanalyze the source to apply 5.3.

Synthetic pipeline tests cover a supported event, source timestamps and JPEGs, contrary actions, all four required predicates in negative/unknown/not-visible states, missing citations, missing evidence and repeated views of one source image. The browser regression checks policy wording, the timestamp, evidence and safe source links. These tests verify the decision path; they do not measure eating-recognition accuracy on independently reviewed real videos. No fine-tuning or new accuracy claim is made.
