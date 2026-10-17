# Local structured VLM backends

The worker can own either its existing llama.cpp process or an optional Qwen2.5-VL Transformers server. Both accept the existing image-and-JSON-schema calls. Candidate verification is bounded to eight images; smaller scene calibration requests remain supported. Original evidence JPEGs retain their source pixels and timestamps. Only the copies sent to inference are resized to the configured pixel budget.

The default remains the previously installed Qwen3-VL-4B GGUF. A pinned 7B GGUF installer and an optional 7B NF4 server are implemented. [Current validation](architecture-v4-validation.md) distinguishes actual model compatibility checks from video performance. The NF4 server has contract tests; its CUDA memory, throughput, model quality and end-to-end deadline have not been validated on this 8 GiB laptop. Four-bit language weights do not imply a four-bit vision encoder, KV cache or activations.

## Existing GGUF backend

`FORTE_MODEL_PATH` and `FORTE_MMPROJ_PATH` select compatible local GGUF weights and their matching vision projector. `scripts/setup_verifier_gguf.py` installs the official runtime-maintainer [ggml-org Qwen2.5-VL-7B GGUF](https://huggingface.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF/tree/508edd0afaa66bb9e9f40587acc2184f02daf1f6) at an immutable revision. It uses exact LFS sizes and SHA-256 values, verifies license provenance, resumes interrupted transfers, and publishes final names atomically. The default Q4_K_M language model is 4,683,072,032 bytes; its matching Q8 projector is 853,119,712 bytes. A larger F16 projector is also available with `--projector f16`.

```powershell
.\.venv\Scripts\python.exe scripts/setup_verifier_gguf.py --connections 8
.\.venv\Scripts\python.exe scripts/setup_verifier_gguf.py --verify-only
```

```powershell
$env:FORTE_VLM_BACKEND = 'llama.cpp'
$env:FORTE_MODEL_PATH = "$PWD\models\vlm\qwen2.5-vl-7b\Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
$env:FORTE_MMPROJ_PATH = "$PWD\models\vlm\qwen2.5-vl-7b\mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf"
$env:FORTE_VLM_CONTEXT = '8192'
.\.venv\Scripts\python.exe scripts/run_local.py
```

Managed llama.cpp receives the configured context limit, schema-constrained response format, one parallel sequence and the existing GPU-layer setting. Its tokenizer/context implementation remains responsible for the exact multimodal context check. The host independently validates every returned object, rejects truncation, enforces image/frame/output bounds and never follows remote redirects.

`FORTE_GPU_LAYERS` controls decoder offload. `FORTE_VLM_CPU_PROJECTOR=1` adds llama.cpp's `--no-mmproj-offload`; paired with `FORTE_GPU_LAYERS=0`, this permits a CPU-only compatibility check while another model owns GPU memory:

```powershell
.\.venv\Scripts\python.exe scripts/check_verifier.py --cpu `
  --model models/vlm/qwen2.5-vl-7b/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf `
  --projector models/vlm/qwen2.5-vl-7b/mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf `
  --output docs/verifier-7b-validation.json
```

The check owns a separate loopback server, submits four synthetic images and validates a constrained response, then stops its owned process. A CPU compatibility result does not establish GPU video throughput or event accuracy.

On 18 October 2026 the pinned Q4_K_M model and Q8 projector were installed and independently rehashed offline. The actual CPU check passed: 38.704 seconds to load and 23.012 seconds for four-image inference, with all four source indices cited. See [the machine-readable result](verifier-7b-validation.json) and [validation scope](architecture-v4-validation.md). The existing 4B verifier remains the default; select the installed 7B assets explicitly with the settings above.

## Optional Qwen2.5-VL-7B NF4 backend

The implementation uses the unquantized official safe-tensor checkpoint and converts supported language linear layers to bitsandbytes NF4 with double quantization at load time. Vision and `lm_head` are excluded from quantization. Prequantized AWQ checkpoints are deliberately rejected by this implementation; AWQ needs a separately validated loader/kernel environment. The official [Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/cc594898137f460bfe9f0759e9844b3ce807cfb5) has approximately 16.6 GB of files before runtime quantization. Keep sufficient disk and host RAM for loading.

Use an isolated Python 3.12 environment; do not replace the working `.venv` or the CPU training environment:

```powershell
py -3.12 -m venv .venv-vlm
.\.venv-vlm\Scripts\python.exe -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-vlm\Scripts\python.exe -m pip install -r requirements.vlm.txt
```

These deliberately pinned versions follow the documented Transformers 4.57.1 API. They are not a claim that the newest package versions were selected or that this combination has passed GPU deployment. PyTorch documents the [2.9.0/0.24.0 CUDA wheel pairing](https://pytorch.org/get-started/previous-versions/); bitsandbytes documents its [platform and CUDA requirements](https://huggingface.co/docs/bitsandbytes/v0.48.1/installation). Select a CUDA wheel supported by the actual GPU/driver rather than changing packages in the working inference environment.

The following is an **explicit online preparation step**, separate from inference. It downloads the public checkpoint at a fixed revision; it was not executed as part of the implementation:

```powershell
.\.venv-vlm\Scripts\hf.exe download Qwen/Qwen2.5-VL-7B-Instruct --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 --local-dir models/qwen2.5-vl-7b
.\.venv-vlm\Scripts\python.exe scripts/serve_vlm.py --model-dir models/qwen2.5-vl-7b --check
```

Retain the model card and download metadata with the snapshot, and record safe-tensor hashes for an audited deployment. `--check` verifies required local files/shards and package availability; it does not load tensor contents, exercise CUDA, attest model identity or prove memory fit.

To let the existing worker own the optional server, set absolute paths:

```powershell
$env:FORTE_VLM_BACKEND = 'transformers'
$env:FORTE_VLM_PYTHON = (Resolve-Path .venv-vlm/Scripts/python.exe).Path
$env:FORTE_TRANSFORMERS_MODEL_DIR = (Resolve-Path models/qwen2.5-vl-7b).Path
$env:FORTE_VLM_CONTEXT = '8192'
$env:FORTE_VLM_MAX_FRAMES = '8'
$env:FORTE_VLM_MAX_PIXELS = '200704'
$env:FORTE_VLM_MIN_PIXELS = '50176'
.\.venv\Scripts\python.exe scripts/run_local.py
```

Start with 256 visual tokens per image (`200704 = 256 × 28 × 28`) and measure the small-object accuracy tradeoff. The default budget is 512 tokens per image (`401408` pixels). The [official processor documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl) describes these image pixel controls. The server also checks the actual processed input-token count plus reserved output tokens against the configured context **before generation**; it never silently truncates the prompt or drops frames. Increasing pixels, frame count or context increases memory pressure.

For a separately managed server:

```powershell
.\.venv-vlm\Scripts\python.exe scripts/serve_vlm.py --model-dir models/qwen2.5-vl-7b --host 127.0.0.1 --port 18080
```

Then set `FORTE_ATTACH_MODEL=1` in the worker's terminal, along with the matching backend/limits. The worker does not terminate an attached process. The server binds to loopback only and accepts embedded JPEG data, not image URLs or file paths.

## Readiness, output guarantees and limits

`GET /health` returns `ok` only after safe-tensor loading, CUDA availability, requested four-bit mode and model-context checks succeed. `GET /v1/runtime` exposes loaded package versions, actual asset SHA-256 hashes, quantization method, configured limits and readiness. `LocalRuntime.metadata()` retains these server fields and supplies a `model_id` fingerprint bound to asset hashes and inference budgets. The managed llama.cpp fingerprint hashes its actual model, projector and server binary. The optional server hashes safe-tensor shards and processor/tokenizer configuration as well as recording package versions. Hashes are cached in process only while file size and timestamps remain unchanged. An attached generic server without identity metadata receives an `unattested:` ID and must not inherit a calibration from a filename match. A caller's calibration key must additionally bind its prompt/pipeline revision; runtime asset identity cannot establish which caller prompt was used.

The optional loader sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, passes `local_files_only=True`, `trust_remote_code=False` and `use_safetensors=True`, and uses one CUDA device without automatic CPU/disk offload. These follow the [offline loading documentation](https://huggingface.co/docs/transformers/v4.57.1/en/installation) and [bitsandbytes configuration](https://huggingface.co/docs/transformers/v4.57.1/en/quantization/bitsandbytes).

Generation uses deterministic decoding with a [LM Format Enforcer JSON-schema token filter](https://github.com/noamgat/lm-format-enforcer), followed by the same bounded host schema validator as llama.cpp. Every schema-allowed rule ID and `not_visible` state is supplied by the caller. Syntax constraints do not establish visual truth. Version 5.2 distinguishes deadline, timeout, HTTP, transport and output errors. Malformed responses and HTTP 502/503 get at most one retry when enough time remains. A failure before full timeline screening fails the job; a later candidate-verification failure preserves existing supported results with explicit incomplete-review coverage. Unsupported output cannot authorize a finding or a clean UI state. See [deadline handling](deadline-fix.md). `max_time` is a generation stopping bound checked between decoding steps, not a hard GPU-kernel deadline; the worker still applies the job deadline and request timeout.

| Setting | Default | Accepted range / meaning |
| --- | --- | --- |
| `FORTE_VLM_BACKEND` | `llama.cpp` | `llama.cpp` or `transformers` |
| `FORTE_VLM_CONTEXT` | 8192 | 2048–32768 tokens |
| `FORTE_VLM_MAX_FRAMES` | 8 | 1–8; candidate minimum is a pipeline concern |
| `FORTE_VLM_MIN_PIXELS` | 50176 | Optional processor minimum; at most maximum |
| `FORTE_VLM_MAX_PIXELS` | 401408 | Inference copy limit; source evidence unchanged |
| `FORTE_VLM_MAX_OUTPUT_TOKENS` | 2048 | 128–4096, smaller than context |
| `FORTE_VLM_PYTHON` | current interpreter | Isolated server interpreter |
| `FORTE_TRANSFORMERS_MODEL_DIR` | `models/qwen2.5-vl-7b` | Complete local checkpoint directory |
| `FORTE_VLM_STARTUP_TIMEOUT` | 120 / 300 | GGUF / Transformers; 30–900 seconds |

The server serializes GPU inference, bounds request/image sizes, rejects missing shards, and returns only generic diagnostic error codes. It does not add model outputs, prompts, embedded image data or video names to access logs. The existing application API authentication and result contract are unchanged.

CPU tests exercise frame limits, offline loader arguments, four-bit readiness, exact context rejection, token-filter attachment, JSON validation, model-process ownership and inference resizing without changing evidence originals. A real acceptance run must separately measure successful model loading, sampled/peak memory, completion time and per-rule accuracy on reviewed held-out videos.
