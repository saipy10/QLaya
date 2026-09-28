# Docker quickstart

Run the SDK without installing Python or PyTorch on your host. For the CPU
quickstart, allow 8 GB of RAM and 10 GB of free disk, with Docker Engine or
Docker Desktop and Compose v2 or newer.

From the repository root:

```bash
docker compose run --build --rm qlaya
```

This builds the checkout, runs the [sample request](https://github.com/saipy10/QLaya/blob/main/examples/docker/request.json)
on CPU and prints JSON covering `choice`, `score` and `noul`. The first request
downloads the selected public Hugging Face checkpoint; no account is needed.
Allow several minutes for its first download.
Weights stay in a named volume. Subsequent runs use `docker compose run --rm qlaya`.

Predictions and confidence still need evaluation on your workload. See the
[benchmark limits](https://github.com/saipy10/QLaya/blob/main/BENCHMARKS.md).

For ARM64 hosts, DGX Spark and Apple Silicon, see
[ARM64 and DGX Spark containers](docker-platforms.md).

## NVIDIA GPU / CUDA

Install a compatible NVIDIA driver and configure Docker with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The GPU image uses PyTorch CUDA 12.8 wheels. Check your GPU's compute capability
and driver against [PyTorch's supported builds](https://pytorch.org/get-started/locally/);
older cards may require a different build. Allow additional disk space for CUDA
layers. VRAM needs depend on the checkpoint, batch size and input length.

```bash
docker compose -f compose.yaml -f compose.cuda.yaml run --build --rm qlaya
```

The override selects GPU `0` and defaults to `QLAYA_DEVICE=cuda`. Set
`QLAYA_GPU_ID` to another host index or UUID. That GPU appears as device `0`
inside the container. Check access without downloading weights:

```bash
docker compose -f compose.yaml -f compose.cuda.yaml run --rm qlaya python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda").cpu())'
```

The sample rejects unavailable CUDA before loading a checkpoint. QLaya can still
fall back to CPU after a memory or inference error, so inspect its warnings.
Rebuild when switching between CPU and CUDA configurations.

The image sets `TORCH_DISABLE_NATIVE_JIT=1`. PyTorch 2.14 otherwise replaces some eager
CUDA ops with Triton kernels that it compiles on the first inference, which needs a C
compiler the slim image does not carry: the container reports healthy and then fails every
request (#365). The stock kernels give the same answers at the same latency. Set the same
variable on a bare-metal install if `predict` fails with `Failed to find C compiler`.

This uses [Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/).
Windows requires Docker Desktop's supported WSL2 GPU setup. Apple MPS,
AMD/ROCm and Intel GPU containers are outside this quickstart; use CPU unless
you configure and validate another backend.

## Configuration

Set Compose variables in your shell, a local `.env` file, or the service's
`environment` block. Don't commit secrets in `.env`. Runtime variables also
work with `docker run -e`; Compose-only settings are identified below.

| Variable | Default | Purpose |
| --- | --- | --- |
| `QLAYA_DEVICE` | `cpu` / `cuda` | Device selected by the base / GPU configuration |
| `QLAYA_CUDA_AMP` | unset (checkpoint's `amp_dtype`) | `fp16` or `bf16` for the CUDA forward. Not cosmetic: the README's threshold section measures bf16 flipping 3 of 864 argmaxes on the parity set where fp16 flips none |
| `QLAYA_CPU_AMP` | unset | `bf16` opts the CPU forward into bf16; anything else leaves it fp32 |
| `QLAYA_MODEL` | `auto` | Router alias: `auto`, `english`, `multilingual`, `typed-decisions` |
| `QLAYA_MODEL_PATH` | unset | Compatible checkpoint path inside the container |
| `QLAYA_REVISION` | unset | Hub commit, branch or tag used for every checkpoint download, or `reviewed` for the reviewed SHAs in `qlaya/revisions.py`; a `revision=` argument still wins |
| `QLAYA_REQUEST_FILE` | bundled request | JSON request path inside the container |
| `OMP_NUM_THREADS` | `4` | CPU threads; keep within available cores |
| `HF_TOKEN` / `HF_TOKEN_FILE` | unset | Optional Hugging Face credential |
| `QLAYA_API_KEY` / `QLAYA_API_KEY_FILE` | unset | **`qlaya-serve` only:** require `Authorization: Bearer <key>` |
| `QLAYA_PORT` | `8000` | **`qlaya-serve` only:** container port, and the host port published for it |
| `HF_HUB_OFFLINE` | `0` | `1` uses only cached checkpoints |
| `HF_HOME` | `/home/qlaya/.cache/huggingface` | Cache path; see mount requirement below |
| `QLAYA_CACHE_VOLUME` | project model cache | **Compose only:** named cache volume |
| `QLAYA_GPU_ID` | `0` | **Compose only:** NVIDIA device index or UUID |
| `QLAYA_TORCH_INDEX` | `cpu` / `cu128` / `cu130` | **Compose build:** PyTorch wheel index |
| `QLAYA_TORCH_VERSION` | `2.14.0` | **Compose build:** pinned PyTorch version |

Compose forwards the runtime variables except `HF_HOME`, which stays aligned
with its fixed cache mount, and except `QLAYA_MPS_AMP_MIN_ROWS`, the MPS row gate,
which no image here can reach because no container here can select MPS.
If overriding `HF_HOME` in `docker run` or your own
Compose file, provide a matching mount writable by UID 10001. Direct Docker
builds select PyTorch with `--build-arg TORCH_INDEX=cu128`; runtime `-e` cannot
change the installed wheel.

```bash
QLAYA_MODEL=english OMP_NUM_THREADS=2 docker compose run --build --rm qlaya

docker build -t qlaya:local .
docker run --rm -e QLAYA_MODEL=english -e OMP_NUM_THREADS=2 \
  -v qlaya-model-cache:/home/qlaya/.cache/huggingface qlaya:local
```

For your own request:

```bash
docker compose run --rm --volume "$PWD/request.json:/inputs/request.json:ro" \
  --env QLAYA_REQUEST_FILE=/inputs/request.json qlaya
```

For a commented configuration with request, checkpoint and secret-file mounts,
see [`compose.example.yml`](https://github.com/saipy10/QLaya/blob/main/compose.example.yml):

```bash
docker compose -f compose.yaml -f compose.example.yml run --build --rm qlaya
```

Add `-f compose.cuda.yaml` before `run` for NVIDIA GPUs. The example is an
override of `compose.yaml`, so cache and image settings stay in one place.

## Secret files

`HF_TOKEN_FILE` reads a mounted UTF-8 file at startup, trims surrounding
whitespace and takes precedence over `HF_TOKEN`. Unreadable, empty or invalid
files stop startup without printing their contents. The file must be readable
by UID 10001. `_FILE` applies only to supported secrets, not every setting.

With `HF_TOKEN_PATH` pointing to an existing host file outside the checkout:

```bash
docker compose run --rm --volume "$HF_TOKEN_PATH:/run/secrets/hf_token:ro" \
  --env HF_TOKEN_FILE=/run/secrets/hf_token qlaya
```

Docker secrets or Kubernetes Secret volumes can supply the same file. Values
are loaded into the process environment at startup; restart after changing a
file. Never use tokens as build arguments or bake them into images. Public
checkpoints need no token.

## Fine-tuned checkpoints

This image runs inference. Fine-tuning happens outside it — the
[fine-tuning notebook](https://github.com/saipy10/QLaya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
runs the whole loop on Kaggle's free 2xT4 GPUs and exports a checkpoint this image can
serve. Background and open questions about the training interface stay in
[#4](https://github.com/saipy10/QLaya/issues/4) and
[#26](https://github.com/saipy10/QLaya/issues/26).

Point `QLAYA_CHECKPOINT_PATH` to an absolute host directory containing
`rl_agent_config.json`, `model.safetensors` and matching tokenizer files:

```bash
docker compose run --rm --volume "$QLAYA_CHECKPOINT_PATH:/models/custom" \
  --env QLAYA_MODEL_PATH=/models/custom qlaya
```

Use a working copy writable by UID 10001 because the loader may update tokenizer
configuration. A LoRA adapter alone is not a complete checkpoint. Leave
`QLAYA_MODEL=auto` when setting `QLAYA_MODEL_PATH`; an explicit alias and local path
are mutually exclusive. The local-path response comes from the Agent and has no
Router `routing` metadata. These settings also work with the CUDA override.
Evaluate fine-tuned checkpoints on held-out examples before relying on them.

## Development and cleanup

Open a Python prompt with `docker compose run --rm qlaya python`. To run the
existing routing/criteria checks and secret-file tests against your checkout
without downloading weights:

```bash
docker compose run --rm --volume "$PWD:/workspace:ro" --workdir /workspace qlaya \
  sh -ec 'python tests/test_router.py; python tests/test_criteria.py; python tests/test_docker_entrypoint.py'
```

Rebuild with `--build` after changing source or the bundled example. The image
runs as UID/GID 10001. New named volumes inherit the image cache directory's
ownership; host directories must be writable by that UID. Keep model caches
writable for tokenizer compatibility updates.

`--rm` removes completed containers. `docker compose down` retains the cache.
To **delete downloaded weights**, run `docker compose down --volumes` using the
same Compose files and `QLAYA_CACHE_VOLUME` setting. The next request downloads
them again; don't remove a cache shared with another project.

## HTTP serving

The image ships `qlaya-serve`, so the same build that runs the one-shot quickstart can
serve the Jev-compatible API. `compose.http.yaml` adds it as a second service and leaves
`qlaya` alone:

```bash
docker compose -f compose.yaml -f compose.http.yaml up --build qlaya-serve
curl -s localhost:8000/health
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' \
  --data @examples/docker/request.json
```

For NVIDIA, add the CUDA override. It repeats the build args and the device reservation
for `qlaya-serve`, because `qlaya-serve` is a separate service and overrides for `qlaya`
never reach it:

```bash
docker compose -f compose.yaml -f compose.http.yaml -f compose.cuda.yaml up --build qlaya-serve
```

`up` keeps the service running in the foreground; `-d` detaches. Weights go to the same
named `model-cache` volume as the quickstart, so serving after a quickstart run starts
with the checkpoints already on disk. Stop with `docker compose ... down`, using the same
Compose files.

The port is published on `127.0.0.1` only. The API has no authentication until
`QLAYA_API_KEY` is set, so set a key before exposing it with
`QLAYA_BIND_ADDRESS=0.0.0.0`, and put a TLS reverse proxy in front for remote clients.
`/health` does not require authentication in either case.

The service has a healthcheck on `/health`. The server preloads before it starts
listening, so with `QLAYA_PRELOAD=1` a healthy container has its checkpoints loaded.
`docker compose ... up -d --wait qlaya-serve` returns once it is healthy.

`/health` reports `device` as the device a resident checkpoint actually computes
on, which is not always what `QLAYA_DEVICE` asked for: a checkpoint that wants a
GPU it cannot get falls back to CPU silently and still answers correctly.
`checkpoint_devices` names each loaded checkpoint, and `device_is_preference` is
`true` only while nothing is resident, so a deployment that quietly lost its GPU
says so instead of echoing its own configuration back.

### Server configuration

These apply to the `qlaya-serve` service only.

| variable | default | effect |
|---|---|---|
| `QLAYA_HOST` | `0.0.0.0` | bind address inside the container |
| `QLAYA_PORT` | `8000` | container port, and the host port published for it |
| `QLAYA_BIND_ADDRESS` | `127.0.0.1` | host address the port is published on |
| `QLAYA_PRELOAD` | `0` | `1` builds every checkpoint at startup instead of on first request |
| `QLAYA_MODELS` | (all) | comma list to preload: `english,multilingual,typed-decisions` |
| `QLAYA_THREADS` | `OMP_NUM_THREADS` | caps torch intra-op threads; keep at or below physical cores |
| `QLAYA_AUTO_TASK` | `0` | `1` lets the router reach `typed-decisions` automatically |
| `QLAYA_MAX_LOADED` | `2` | Checkpoints kept resident; `QLAYA_AUTO_TASK` makes a third reachable on demand, and a cap below what routing chooses rebuilds one per switch |
| `QLAYA_MAX_CONCURRENT` | `16` | requests admitted at once; later ones get `503` (a value that does not parse, or is not positive, falls back to `16`) |
| `QLAYA_LOG_LEVEL` | `info` | uvicorn log level |
| `QLAYA_API_KEY` | (none) | when set, requires `Authorization: Bearer <key>` |
| `QLAYA_MAX_TOKEN_BUDGET` | `8192` | cap on per-request `max_len` and `head_max_len` overrides |

`QLAYA_PRELOAD` defaults to `0` here rather than the package default of `1`, because
preloading makes the first boot download all three checkpoints. Set it to `1` for a
long-running deployment so the first request does not pay for the build.

`QLAYA_PORT` sets both the published host port and the port the server binds, so the two
cannot drift. Change one place to move the service:

```bash
QLAYA_PORT=9000 docker compose -f compose.yaml -f compose.http.yaml up --build qlaya-serve
```

### Bearer token from a file

`QLAYA_API_KEY_FILE` is read once at startup, moved into `QLAYA_API_KEY`, and the `_FILE`
variable is removed before the server execs. Prefer this to putting the key in the
environment:

```bash
docker compose -f compose.yaml -f compose.http.yaml run --rm \
  --volume "$PWD/laya_api_key:/run/secrets/laya_api_key:ro" \
  -e QLAYA_API_KEY_FILE=/run/secrets/laya_api_key \
  --service-ports qlaya-serve
```

