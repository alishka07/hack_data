# Architecture 4.0 validation, 18 October 2026

The implementation passed software and local model-operation checks. Detection accuracy, per-rule calibration and unseen-camera performance remain unfinished. Passing API checks or detecting more objects does not establish better violation detection.

## Executed checks

- The full Python regression suite passed: **801 tests in 28.59 seconds**. One existing Starlette/httpx deprecation warning remains.
- Headless Chrome's synthetic UI regression passed, including authenticated deferred JPEGs, source seeking, review decisions, dense pagination, coverage, legacy results and safe text rendering.
- Python compilation and JavaScript syntax checks passed.
- Actual ByteTrack recovered a low-score detection and marked reappearance after a missing sample as unknown continuity. Pose/object shape, fixed-query and source-reference checks passed.
- Actual local RTMPose decoded 17 COCO keypoints. Actual local OWLv2 loaded offline and inspected synthetic pixels. These checks establish callable model compatibility, not localization accuracy.
- The real authenticated API accepted a generic-named MOV, rejected unauthenticated/corrupt/busy requests, completed polling and served authenticated evidence. The isolated test API/worker were stopped afterward; their attached model server was not owned by the test.

## Resident 4B model and complete perception stack

The video runs used the already resident local Qwen3-VL-4B Q4_K_M server, genuine ByteTrack, RTMPose ONNX and OWLv2 CPU. Optional CPU weights load within the job budget. The verifier's initial load is excluded because it was already serving; these are warm-workstation operation measurements, not a cold-start or capacity benchmark.

| Check | Source duration | Observed completion | Result |
| --- | ---: | ---: | --- |
| Augmented appearance clip, perception only | 25.933 s | 29.734 s | 131 source samples, 137 pose records and 454 detection hypotheses; all perception capabilities enabled |
| Short original, direct pipeline | 11.633 s | 96.528 s | Completed; four review items, zero confirmed findings |
| Longest original, final direct pipeline | 142.533 s | 237.602 s | Completed within 300 seconds; two review items, zero confirmed findings |
| Short original, actual upload/API | 11.633 s | 66.531 s | All technical checks passed; review required, zero confirmed findings |

The short API check validated 97 source-linked boxes and three declared authenticated JPEGs. The direct short/long checks decoded all 59/714 declared evidence JPEGs respectively. No real confirmed positive finding was emitted, so original-dimension *positive-event* evidence quality has only synthetic pipeline validation. The short API run sampled a whole-device maximum of 6,397 MiB out of 8,151 MiB; this includes other processes and is not a true instantaneous or model-only peak.

The final long run inspected 714 person/motion frames, 257 pose-sampling frames and 29 object frames, generated 82 temporal proposals, and checked 33 screening windows. Perception took 82.354 seconds within its 90-second limit; VLM work took 123.220 seconds within its 150-second target. Decode/cache and evidence/output also met their stage targets. Repeated source footage, workstation load and model caches affect timing; these runs do not guarantee the deadline on a different video or candidate flood.

Raw aggregate reports: [short](architecture-v4-shortest.json), [long](architecture-v4-longest.json), [API](architecture-v4-api.json). Private observations, verification attempts and source JPEGs remain under ignored `data/` and `tmp/`.

## Quality result and fine-tuning implications

These runs cover **two unique original videos**. Repeating the short source through the API is not a third independent quality example. The short reference is positive and was still missed by the jury verdict; the long negative reference returned no finding. This is TP=0, FP=0, TN=1, FN=1 on previously inspected weakly labeled footage. It is not an independent evaluation or a per-rule accuracy estimate.

The short API run created four review items, with action verification still uncertain and no context-verification calls. That locates unresolved processing in the visible-action stage; it does not establish the correct rule or event boundary. Review the actual event intervals, action crops, visibility and proposed source anchors before treating these outputs as training targets. No augmented or model-generated answer was promoted to ground truth.

[The full data audit](finetuning-analysis.md) hashed and sampled all 402 files. Its 40 original hash groups have binary labels; detector boxes, pose visibility, scene polygons, event intervals and source-grounded rule predicates remain unreviewed or absent. Fine-tuning and empirical decision calibration need those respective targets. No calibration artifact or trained adapter was created from the binary workbook alone.

## 7B compatibility scope

The official runtime-maintainer Qwen2.5-VL-7B Q4_K_M model and matching Q8 projector are installed at pinned revision `508edd0afaa66bb9e9f40587acc2184f02daf1f6`. The files total 5,536,191,744 bytes. Download-time SHA-256 verification and a separate offline rehash both passed; the local model card, license and manifest are retained with the assets.

An actual isolated CPU-only llama.cpp server loaded these assets in **38.704 seconds**, then answered a four-image synthetic visual question with schema-constrained output in **23.012 seconds**. The response correctly reported visible colored squares and cited all four supplied frame indices. Total check time was **62.575 seconds**, including cleanup. The owned test server stopped afterward. [The recorded 7B report](verifier-7b-validation.json) includes the actual asset hashes, runtime identity, zero decoder GPU layers and CPU projector setting.

This establishes local weight/projector, multimodal request and constrained-output compatibility. The video runs above used the resident 4B verifier. Neither those timings, the GPU sample nor this synthetic CPU check establishes 7B GPU fit, complete video latency or rule accuracy. The optional Transformers NF4 backend has executable inference/server code and synthetic contract tests; its separate safe-tensor checkpoint and CUDA loading have not been exercised.
