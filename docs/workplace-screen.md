# Workstation departure with the computer screen left on

Pipeline 5.5 adds `workstation_departure_screen_left_on`, displayed as **Workstation left with computer screen on**. This is the explicitly recorded [user-provided workplace policy](../rules/sources/workplace-screen.md), not a newly asserted public-law requirement. The rule applies independently of active client intake or an unfinished cash transaction.

The pipeline locates laptops and desktop monitors with an additional OWLv2 hardware query and the source-linked localization schema. Per-video calibration can propose a computer region; verification can inspect source-pixel computer crops while retaining full-scene before/after images. Hardware detection alone does not establish screen power or session locking.

A supported event requires all five source-cited checks: an identified workstation computer, an actual workstation departure, an unattended workstation afterwards, the same screen visibly on afterwards, and a readable sequence linking these conditions. Departure citations must cover at least two distinct source images and times. The computer, unattended workstation and illuminated display must share a cited post-departure source image. Multiple crops of one image cannot establish departure.

The primary event time and JPEG come from the post-departure state; additional evidence includes the earlier workstation view. Material source-clock gaps or a scene cut across the sequence prevent confirmation. An empty-chair still, standing/reaching, a cropped-out operator, a replacement attendant, a different computer, glare/reflection, or an unreadable display cannot establish the event. A lit lock screen is still on under the user's policy; no login state is inferred.

The supplied example screenshot received a local CPU OWLv2 smoke check: the new hardware query localized the laptop at the existing threshold. That checks device localization on one example only. A still image cannot establish the full departure sequence. Synthetic tests cover the ordered event, source consistency, genuine evidence JPEGs, negative/unknown/missing checks, display ambiguity and the lit-lock-screen case. Browser checks cover the rule label, policy reference, timestamp, evidence and probability. These checks do not establish recognition accuracy across cameras or independently reviewed videos.

Older completed results retain their original interpretation. Refresh and reanalyze the source video to apply the new rule. The rule's Results probability remains an uncalibrated model estimate.
