# Food as personal belongings at the cashier work table

Pipeline 5.4 applies the user's revised rule: identifiable food or a snack on the operating cashier work table is personal belongings at the workplace and counts as a violation. Eating, chewing, food handling and a consumption sequence are not required. This is the recorded [user-provided workplace policy](../rules/sources/workplace-eating.md), not a newly asserted public-law clause.

The active rule is `personal_food_at_workstation`, displayed as **Personal belongings at workplace (food)**. It requires identified food, a visually supported cashier work-surface location and visible work context. All required checks must reference a common source image that is also cited as evidence. A single readable source image is sufficient. The primary event timestamp and evidence frame show the food and its workplace relationship.

Unknown objects, unknown cup contents, empty wrappers alone, medicine, cash and ordinary work equipment do not establish identifiable food. Food outside the cashier work surface or in a break area does not meet this rule. The separate review of other personal belongings retains its existing scope and exceptions.

The former `cashier_eating_at_workstation` rule is disabled for new analyses. Previous saved results are not relabelled; reanalyze the video to apply the revised rule. The Results probability column remains an uncalibrated model estimate, and all applicability, source-citation and evidence checks must pass before a violation is emitted.

Synthetic tests cover food present without eating, a single image, presence at the start of a clip, false-positive alternatives, unknown/negative/missing checks, source intersections and evidence, policy provenance, and the retired eating rule. Browser checks cover the personal-belongings label, policy reference, timestamp, JPEG and probability. These checks establish the implementation behavior; recognition accuracy on independently reviewed real footage remains unmeasured.
