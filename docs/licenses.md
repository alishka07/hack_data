# License text audit and retained notices

Architecture 4.0 adds optional runtime distributions and pretrained assets outside the original 28-distribution audit below. Installed package metadata identifies `trackers` 2.6.1 as Apache-2.0, `supervision` 0.26.1 as MIT and ONNX Runtime 1.23.2 as MIT. RTMPose and OWLv2 setup retains upstream Apache-2.0 provenance and checked asset hashes under `models/perception`; the 7B GGUF installer retains its pinned Apache-2.0 model card, canonical license text and checksums under `models/vlm/qwen2.5-vl-7b`. CPU Torch/Transformers and optional NF4 dependencies are isolated. These additions are documented in [the current architecture](architecture-v4.md) and are not represented as covered by the older complete-distribution license-text audit.

The remaining sections are the historical 3.0/baseline audit.

Pipeline 3.0 adds a separate, setup-only CPU YOLOX Nano ONNX asset. Its release URL, size, SHA-256 and upstream provenance are documented in [perception.md](perception.md); the Apache-2.0 text is retained at [yolox-0.1.1rc0/LICENSE.txt](third-party-licenses/yolox-0.1.1rc0/LICENSE.txt). These assets are additional to the original 28-distribution audit below. No production Python dependency was added for YOLOX inference; it uses the existing OpenCV DNN backend. Optional isolated tracker-evaluation packages are outside the application runtime audit and are not required to run the service.

Reviewed on 17 October 2026 against the installed Windows Python 3.12.14 environment. The 28 application/test distributions in [requirements.lock.txt](../requirements.lock.txt) have primary licenses permitting commercial use under their terms. This conclusion follows actual license grants and conditions, including the official MPL 2.0 text for certifi, rather than package metadata alone. Bundled binaries have additional conditions and the specific evidence gaps listed below.

[license-audit.json](license-audit.json) records every inspected installed path, byte length, full SHA-256, retained copy, primary grant excerpt and line number. All discovered package LICENSE/LICENCE/COPYING/NOTICE files are retained byte-for-byte under [third-party-licenses](third-party-licenses/), including NumPy's nested notices and OpenCV's complete third-party notice. The table fingerprints below are the first 16 hex characters; use the manifest for full hashes. pip is bootstrap tooling and is outside the 28-distribution lock/audit scope.

## Primary grants and conditions

These are conclusions from the retained text; the text itself controls.

- **M — MIT:** permission expressly includes use, modification, distribution, sublicensing and selling copies. Retain the copyright and permission notice in copies or substantial portions.
- **B — BSD-3-Clause:** use and source/binary redistribution, with or without modification, are permitted. Retain copyright, conditions and disclaimer in the required source/binary materials; do not imply endorsement.
- **B2 — packaging's BSD-2-Clause option:** packaging's root notice offers either Apache-2.0 or BSD-2-Clause. The reviewed BSD text permits use and redistribution, requiring copyright, conditions and disclaimer retention.
- **A — Apache-2.0:** sections 2 and 3 grant copyright/patent permissions, including distribution and sale-related patent rights. Section 4 requires the license, preservation of relevant notices, marking changed files, and NOTICE attribution when supplied. Trademark rights are separate.
- **P — Python/PSF:** the retained Python terms grant royalty-free use, modification and distribution, with PSF copyright/license retention and a summary of changes when applicable. Historical Python notices are retained too. et_xmlfile additionally carries MIT terms for its own code.
- **C — certifi/MPL-2.0:** the installed file contains Mozilla's MPL notice for the CA bundle, with the full grant incorporated by reference. Official [MPL sections 2.1 and 3.1–3.4](https://www.mozilla.org/en-US/MPL/2.0/) permit exploitation and sale under file-level distribution conditions; retain notices and provide covered source when required. [Mozilla FAQ Q5/Q8/Q11](https://www.mozilla.org/en-US/MPL/2.0/FAQ/) confirms company use and explains covered-source/proprietary-work boundaries.
- **N/O:** primary NumPy and OpenCV-wrapper grants permit commercial use, but do not replace the bundled component terms reviewed below.

| Distribution / installed version | License text and condition code | Actual primary file / retained evidence | SHA-256 prefix |
|---|---|---|---|
| annotated-doc 0.0.5 | MIT / M | [installed](../.venv/Lib/site-packages/annotated_doc-0.0.5.dist-info/licenses/LICENSE), [retained](third-party-licenses/annotated-doc-0.0.5/annotated_doc-0.0.5.dist-info/licenses/LICENSE) line 5 | fff170779a6acbf6 |
| annotated-types 0.8.0 | MIT / M | [installed](../.venv/Lib/site-packages/annotated_types-0.8.0.dist-info/licenses/LICENSE), [retained](third-party-licenses/annotated-types-0.8.0/annotated_types-0.8.0.dist-info/licenses/LICENSE) line 5 | fe1049884b1a0d93 |
| anyio 4.15.1 | MIT / M | [installed](../.venv/Lib/site-packages/anyio-4.15.1.dist-info/licenses/LICENSE), [retained](third-party-licenses/anyio-4.15.1/anyio-4.15.1.dist-info/licenses/LICENSE) line 5 | 5361ac9dc58f2ef5 |
| certifi 2026.7.22 | MPL-2.0 / C | [installed](../.venv/Lib/site-packages/certifi-2026.7.22.dist-info/licenses/LICENSE), [retained](third-party-licenses/certifi-2026.7.22/certifi-2026.7.22.dist-info/licenses/LICENSE) line 15 | e93716da6b9c0d5a |
| click 8.5.0 | BSD-3-Clause / B | [installed](../.venv/Lib/site-packages/click-8.5.0.dist-info/licenses/LICENSE.txt), [retained](third-party-licenses/click-8.5.0/click-8.5.0.dist-info/licenses/LICENSE.txt) line 3 | 9a8ad106a394e853 |
| colorama 0.4.6 | BSD-3-Clause / B | [installed](../.venv/Lib/site-packages/colorama-0.4.6.dist-info/licenses/LICENSE.txt), [retained](third-party-licenses/colorama-0.4.6/colorama-0.4.6.dist-info/licenses/LICENSE.txt) line 4 | cac35c02686e5d04 |
| et_xmlfile 2.0.0 | MIT plus retained PSF/Python notices / P | [installed](../.venv/Lib/site-packages/et_xmlfile-2.0.0.dist-info/LICENCE.rst), [retained](third-party-licenses/et-xmlfile-2.0.0/et_xmlfile-2.0.0.dist-info/LICENCE.rst) line 6 | 0c84bb42f5d367e5 |
| fastapi 0.143.3 | MIT / M | [installed](../.venv/Lib/site-packages/fastapi-0.143.3.dist-info/licenses/LICENSE), [retained](third-party-licenses/fastapi-0.143.3/fastapi-0.143.3.dist-info/licenses/LICENSE) line 5 | 4ec89ffc81485b97 |
| h11 0.16.0 | MIT / M | [installed](../.venv/Lib/site-packages/h11-0.16.0.dist-info/licenses/LICENSE.txt), [retained](third-party-licenses/h11-0.16.0/h11-0.16.0.dist-info/licenses/LICENSE.txt) line 5 | 37db5bb85926db28 |
| httpcore 1.0.9 | BSD-3-Clause / B | [installed](../.venv/Lib/site-packages/httpcore-1.0.9.dist-info/licenses/LICENSE.md), [retained](third-party-licenses/httpcore-1.0.9/httpcore-1.0.9.dist-info/licenses/LICENSE.md) line 4 | fdcb59154c74cbab |
| httpx 0.28.1 | BSD-3-Clause / B | [installed](../.venv/Lib/site-packages/httpx-0.28.1.dist-info/licenses/LICENSE.md), [retained](third-party-licenses/httpx-0.28.1/httpx-0.28.1.dist-info/licenses/LICENSE.md) line 4 | 4ec59d544f12b5f5 |
| idna 3.20 | BSD-3-Clause / B | [installed](../.venv/Lib/site-packages/idna-3.20.dist-info/licenses/LICENSE.md), [retained](third-party-licenses/idna-3.20/idna-3.20.dist-info/licenses/LICENSE.md) line 6 | 1a9a4f0e3d479a27 |
| iniconfig 2.3.1 | MIT / M | [installed](../.venv/Lib/site-packages/iniconfig-2.3.1.dist-info/licenses/LICENSE), [retained](third-party-licenses/iniconfig-2.3.1/iniconfig-2.3.1.dist-info/licenses/LICENSE) line 5 | 3409fa91f7ace557 |
| numpy 2.5.4 | BSD-3-Clause plus bundled licenses / N | [installed](../.venv/Lib/site-packages/numpy-2.5.4.dist-info/licenses/LICENSE.txt), [retained](third-party-licenses/numpy-2.5.4/numpy-2.5.4.dist-info/licenses/LICENSE.txt) line 4 | a804dff0ead9fadc |
| opencv-python-headless 5.0.0.93 | MIT wrapper; Apache-2.0 OpenCV; bundled licenses / O | [installed](../.venv/Lib/site-packages/opencv_python_headless-5.0.0.93.dist-info/LICENSE.txt), [retained](third-party-licenses/opencv-python-headless-5.0.0.93/opencv_python_headless-5.0.0.93.dist-info/LICENSE.txt) line 5 | edef0fac1eb08d29 |
| openpyxl 3.1.5 | MIT / M | [installed](../.venv/Lib/site-packages/openpyxl-3.1.5.dist-info/LICENCE.rst), [retained](third-party-licenses/openpyxl-3.1.5/openpyxl-3.1.5.dist-info/LICENCE.rst) line 6 | 0c84bb42f5d367e5 |
| opentelemetry-api 1.45.1 | Apache-2.0 / A | [installed](../.venv/Lib/site-packages/opentelemetry_api-1.45.1.dist-info/licenses/LICENSE), [retained](third-party-licenses/opentelemetry-api-1.45.1/opentelemetry_api-1.45.1.dist-info/licenses/LICENSE) line 66 | c71d239df91726fc |
| packaging 26.3 | Apache-2.0 OR BSD-2-Clause (BSD option reviewed) / B2 | [installed](../.venv/Lib/site-packages/packaging-26.3.dist-info/licenses/LICENSE.BSD), [retained](third-party-licenses/packaging-26.3/packaging-26.3.dist-info/licenses/LICENSE.BSD) line 4 | b70e7e9b742f1cc6 |
| pluggy 1.7.0 | MIT / M | [installed](../.venv/Lib/site-packages/pluggy-1.7.0.dist-info/licenses/LICENSE), [retained](third-party-licenses/pluggy-1.7.0/pluggy-1.7.0.dist-info/licenses/LICENSE) line 5 | d6b65e6c213a5d0b |
| pydantic 2.12.5 | MIT / M | [installed](../.venv/Lib/site-packages/pydantic-2.12.5.dist-info/licenses/LICENSE), [retained](third-party-licenses/pydantic-2.12.5/pydantic-2.12.5.dist-info/licenses/LICENSE) line 5 | a9e186f3ca16b5ee |
| pydantic_core 2.41.5 | MIT / M | [installed](../.venv/Lib/site-packages/pydantic_core-2.41.5.dist-info/licenses/LICENSE), [retained](third-party-licenses/pydantic-core-2.41.5/pydantic_core-2.41.5.dist-info/licenses/LICENSE) line 5 | fbe7f615f18d135c |
| Pygments 2.21.0 | BSD-3-Clause / B | [installed](../.venv/Lib/site-packages/pygments-2.21.0.dist-info/licenses/LICENSE), [retained](third-party-licenses/pygments-2.21.0/pygments-2.21.0.dist-info/licenses/LICENSE) line 4 | a9d66f1d526df02e |
| pytest 9.1.1 | MIT / M | [installed](../.venv/Lib/site-packages/pytest-9.1.1.dist-info/licenses/LICENSE), [retained](third-party-licenses/pytest-9.1.1/pytest-9.1.1.dist-info/licenses/LICENSE) line 5 | ca836a5f9ecca3b2 |
| python-multipart 0.0.32 | Apache-2.0 / A | [installed](../.venv/Lib/site-packages/python_multipart-0.0.32.dist-info/licenses/LICENSE.txt), [retained](third-party-licenses/python-multipart-0.0.32/python_multipart-0.0.32.dist-info/licenses/LICENSE.txt) line 67 | cfc7749b96f63bd3 |
| starlette 1.8.0 | BSD-3-Clause / B | [installed](../.venv/Lib/site-packages/starlette-1.8.0.dist-info/licenses/LICENSE.md), [retained](third-party-licenses/starlette-1.8.0/starlette-1.8.0.dist-info/licenses/LICENSE.md) line 4 | dcb95677a0224024 |
| typing_extensions 4.16.0 | PSF-2.0 plus historical Python notices / P | [installed](../.venv/Lib/site-packages/typing_extensions-4.16.0.dist-info/licenses/LICENSE), [retained](third-party-licenses/typing-extensions-4.16.0/typing_extensions-4.16.0.dist-info/licenses/LICENSE) line 82 | 3b2f81fe21d181c4 |
| typing-inspection 0.4.4 | MIT / M | [installed](../.venv/Lib/site-packages/typing_inspection-0.4.4.dist-info/licenses/LICENSE), [retained](third-party-licenses/typing-inspection-0.4.4/typing_inspection-0.4.4.dist-info/licenses/LICENSE) line 5 | 804b59b25f2c31bd |
| uvicorn 0.54.0 | BSD-3-Clause / B | [installed](../.venv/Lib/site-packages/uvicorn-0.54.0.dist-info/licenses/LICENSE.md), [retained](third-party-licenses/uvicorn-0.54.0/uvicorn-0.54.0.dist-info/licenses/LICENSE.md) line 4 | efe1acf3e62fb99c |

## NumPy wheel: bundled and nested notices

The installed main LICENSE.txt explicitly identifies OpenBLAS (BSD-3-Clause), LAPACK (BSD-3-Clause-Open-MPI) and the GCC runtime (GPL-3.0-or-later WITH GCC-exception-3.1) bundled inside numpy.libs/libscipy_openblas*.dll. BSD grants permit use and redistribution subject to notice/disclaimer retention and no endorsement.

The full GCC runtime exception is present in that same file. Its section 1 permits eligible compiled combinations of independent modules with the runtime under chosen terms; the exception's introductory text expressly includes proprietary programs. This is additional permission, not removal of the GCC runtime's own license or all redistribution duties.

Nested retained notices cover libdivide (zlib, expressly permitting commercial applications), pythoncapi-compat (0BSD), Highway (Apache-2.0 OR BSD-3-Clause, with a CC0 random header), Dragon4/PCG64/SFC64 (MIT), x86-simd-sort/SVML/pocketfft/LAPACK/masked-array/Philox (BSD terms), random support (BSD/NCSA alternatives and included MIT notices), and SplitMix64 (public-domain dedication/CC0 reference). Their complete text and hashes are in the manifest; no top-level BSD label is used to conceal these components.

## OpenCV wheel: wrapper, native library and codecs

The installed cv2/LICENSE.txt is MIT for the Python packaging wrapper. cv2/LICENSE-3RD-PARTY.txt starts with Apache-2.0 for the OpenCV native library and includes the complete additional notices. The actual installed build reports: GUI NONE; FFmpeg enabled; JPEG/libjpeg-turbo; WebP; PNG; TIFF; JPEG2000/OpenJPEG; built-in CLapack; Protobuf; Flatbuffers; and Intel IPP 2026.0.0. AVIF is reported disabled. Selected exact build metadata is retained in the JSON audit.

| Native component / notice section | Inspected grant and concrete conditions |
|---|---|
| FFmpeg, notice lines 243 onward | LGPL-2.1 text permits use and redistribution; distributing the library/linked binaries requires the applicable source, notice and relinking/replacement conditions. Preserve the full notice. It is distinct from the external GPL FFmpeg executable. |
| libwebp/libwebpmux, lines 2144 onward | BSD-3-Clause use/redistribution grant; retain notices/disclaimer; no endorsement. |
| libjpeg-turbo, lines 2459 onward | IJG, BSD and zlib terms are described together. Binary product documentation must include the IJG attribution below; retain applicable notices, do not imply endorsement, and mark source changes where required. |
| PNG, TIFF, OpenJPEG, zlib and CLapack sections | Their retained texts permit use/distribution with their respective attribution, origin, notice and disclaimer conditions. CLapack's Windows-specific section begins at line 3568. |
| Protobuf and Flatbuffers sections | Retained BSD/Apache grants permit commercial use with the applicable notices and Apache redistribution conditions. |
| Intel IPP, lines 2953 onward | Intel Simplified Software License text expressly permits use and redistribution of unmodified binary software, retaining terms/copyright and avoiding endorsement. It prohibits modification and reverse engineering. This is a proprietary permission, not an open-source label. |
| MLAS and included feature libraries | Retained MIT/BSD grants allow commercial use subject to their respective notice/endorsement conditions. |

**Required IJG attribution:** This software is based in part on the work of the Independent JPEG Group.

The third-party notice is shared across platforms and names Qt/macOS/Linux libraries as well. Their presence in the notice is not treated as evidence that this Windows headless wheel contains them. The Intel IPP notice/version mismatch is recorded below rather than silently certified.

## Local model/runtime and external executables

The following actual notice files were read and retained. Original absolute/relative installed paths are recorded in the manifest.

| Component | Inspected license evidence | Retained file | SHA-256 prefix |
|---|---|---|---|
| llama.cpp-b11707 | MIT | [llama-LICENSE.txt](third-party-licenses/llama.cpp-b11707/llama-LICENSE.txt) | 94f29bbed6a22c35 |
| llama.cpp-b11707 | Apache-2.0 WITH LLVM-exception and retained legacy notices | [LICENSE-LLVM-OpenMP](third-party-licenses/llama.cpp-b11707/LICENSE-LLVM-OpenMP) | fdad1758a9e1f9d5 |
| qwen3-vl-4b-pinned | Model card declares Apache-2.0 | [Qwen-model-card.md](third-party-licenses/qwen3-vl-4b-pinned/Qwen-model-card.md) | 3d50ed633b88c422 |
| qwen3-vl-4b-pinned | Apache-2.0 code notice; separate provenance from model card | [Qwen-code-LICENSE](third-party-licenses/qwen3-vl-4b-pinned/Qwen-code-LICENSE) | c71d239df91726fc |
| external ffmpeg/ffprobe 9.0.1-full_build-www.gyan.dev | GPL-3.0-or-later; README records GPL v3 and source commit bf1b838f2a | [LICENSE](third-party-licenses/ffmpeg-9.0.1-gyan/LICENSE) | 8ceb4b9ee5adedde |
| external ffmpeg/ffprobe 9.0.1-full_build-www.gyan.dev | GPL-3.0-or-later; README records GPL v3 and source commit bf1b838f2a | [README.txt](third-party-licenses/ffmpeg-9.0.1-gyan/README.txt) | 9d734beeeafa8b18 |

- **Qwen model:** the retained pinned model card's front matter declares Apache-2.0. The [official card at revision 1cd86afb9a95c410a6038ab3b40d8b578c892266](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/blob/1cd86afb9a95c410a6038ab3b40d8b578c892266/README.md) and exact model-file hashes are referenced in [README provenance](../README.md#modelruntime-provenance). Apache's grant permits commercial use under its conditions. The retained Qwen code LICENSE supplies Apache text; it is separately identified because a standalone model-repository LICENSE was unavailable.
- **llama.cpp b11707:** the actual ggml MIT notice expressly includes selling copies and requires retaining its notice. The actual libomp notice includes Apache-2.0 with LLVM exceptions and historical LLVM notices; preserve the complete file.
- **External FFmpeg/ffprobe:** both resolve to the installed Gyan 9.0.1 full build. Its actual README says GPL v3 and records FFmpeg source commit bf1b838f2a. The executable's local -L output confirms GPL-3.0-or-later with GPL/version3 build options. The retained GPL text's section 4 allows charging for copies; section 6 imposes corresponding-source duties when conveying binaries. Local execution and binary redistribution are different actions.
- **NVIDIA DLLs:** the local runtime contains cudart64_13.dll, cublas64_13.dll and cublasLt64_13.dll. [Official CUDA EULA sections 1.1 and 2.6](https://docs.nvidia.com/cuda/eula/index.html) permit SDK use and identify versioned runtime/cuBLAS files as distributable within qualifying applications, subject to distribution conditions and NVIDIA-GPU scope. These are proprietary terms; an exact delivered-version EULA was not in the local runtime bundle.

## Remaining precise gaps

- Installed OpenCV reports Intel IPP 2026.0.0, while its bundled third-party notice labels Intel IPP 2021 Update 10 and October 2022 license terms; exact-version notice coverage is not independently established.
- The llama.cpp runtime bundle includes cudart64_13.dll, cublas64_13.dll and cublasLt64_13.dll without a local NVIDIA EULA copy. Official current CUDA terms were checked; exact delivered-version terms and packaging are not fully archived.
- FFmpeg vendor README identifies source commit bf1b838f2a. Complete corresponding source/build inputs for all enabled external codecs are not archived here; binary redistribution compliance is unfinished.
- The Qwen pinned model card declares Apache-2.0, but no standalone model-repository LICENSE was available during planning. The retained Qwen code Apache notice is a separate source and is not relabeled as the missing model LICENSE.
- This audit inspects declared license/notice files, not every statically linked binary object or patent claim. Platform-specific third-party notices are not treated as proof that every named component is present on Windows.

Commercial-use permission was found for the reviewed primary package licenses and the stated native-component grants. This audit does not declare a future redistributed installer fully compliant: corresponding-source packaging, exact binary-version notices and NVIDIA terms must match what is actually shipped. Current source-only repository delivery does not bundle the ignored model/runtime/virtual-environment directories.

Python itself, Windows, the NVIDIA driver and any system redistributables are outside the 28 Python-distribution audit. Their platform notices and terms remain separate.
