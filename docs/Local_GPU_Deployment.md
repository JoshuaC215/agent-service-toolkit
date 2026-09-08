# Running local models on local hardware

You can run the LLM powering the agent service on your own local hardware by self-hosting an OpenAI-compatible inference server with [vLLM](https://docs.vllm.ai/) or [SGLang](https://docs.sglang.ai/). The service connects to it through its built-in `openai-compatible` provider, so no code changes are required.

Because this is an agent service, use a model with tool calling and start the server with tool-call parsing enabled — otherwise agents that call tools (including the default `research-assistant`) will not work.

## 1. Start an inference server on your local hardware

To deploy on local hardware — for example, an **Intel Arc Pro GPU** — you can use either vLLM or SGLang's Intel XPU images. Both serve `Qwen/Qwen3-4B-Instruct-2507` with tool calling enabled and expose an OpenAI-compatible API. Pick one.

### Option A: vLLM

This example uses the vLLM Intel XPU image (`vllm/vllm-openai-xpu`):

```sh
docker run -it --rm \
  --name vllm-service \
  --privileged \
  --net=host \
  --device=/dev/dri \
  --shm-size=8g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  vllm/vllm-openai-xpu:v0.27.1 \
  Qwen/Qwen3-4B-Instruct-2507 \
  --dtype=bfloat16 \
  --max-model-len=8192 \
  --port=8000 \
  --host=0.0.0.0 \
  --enforce-eager \
  --trust-remote-code \
  --gpu-memory-util=0.9 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser=hermes
```

Notes:

- `--device=/dev/dri` exposes the Intel GPU to the container; `--net=host` publishes the server on the host's port `8000`.
- `--enable-auto-tool-choice` and `--tool-call-parser=hermes` enable OpenAI-style tool calling.
- Adjust `--max-model-len`, `--gpu-memory-util`, and the model to fit your GPU.

This exposes an OpenAI-compatible API at `http://localhost:8000/v1`.

### Option B: SGLang

This example uses an Intel XPU SGLang image (`sglang-xpu`):

```sh
docker run -it --rm \
  --privileged \
  --ipc=host \
  --network=host \
  --user root \
  --group-add "$(getent group video | cut -d: -f3)" \
  --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path \
  -v /dev/shm:/dev/shm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e ZE_AFFINITY_MASK=0,1 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:* \
  sglang-xpu:latest \
  /bin/bash -c \
  'sglang serve --model-path Qwen/Qwen3-4B-Instruct-2507 --trust-remote-code --disable-overlap-schedule --device xpu --host 0.0.0.0 --tp 2 --attention-backend intel_xpu --page-size 128 --tool-call-parser qwen --grammar-backend xgrammar'
```

Notes:

- `--device /dev/dri` and the `/dev/dri/by-path` mount expose the Intel GPUs; `--group-add video` grants the container access to them. `--network=host` publishes the server on the host.
- `-e ZE_AFFINITY_MASK=0,1` and `--tp 2` select and split the model across both GPUs (adjust the mask and tensor-parallel size to match the number of GPUs you want to use).
- `--tool-call-parser qwen` enables OpenAI-style tool calling for the Qwen model family.
- SGLang's default port is `30000`, so this exposes an OpenAI-compatible API at `http://localhost:30000/v1`.

### Other hardware

The examples above use Intel XPU images. If you're deploying on different hardware (e.g. NVIDIA or AMD GPUs), swap in the image and device flags for your platform — the rest of the setup (tool-call parser, `.env` configuration) stays the same. See:

- [GPU - vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#set-up-using-docker)
- [Hardware Platforms - SGLang Documentation](https://docs.sglang.io/docs/hardware-platforms/overview)

## 2. Point the agent service at it

Set the following in your `.env` file (use port `8000` for vLLM or `30000` for SGLang, matching whichever server you started in step 1):

```sh
COMPATIBLE_MODEL=Qwen/Qwen3-4B-Instruct-2507   # must match the model the server serves
COMPATIBLE_BASE_URL=http://localhost:8000/v1
COMPATIBLE_API_KEY=sk-not-needed               # any non-empty value; local servers don't check it
```

`COMPATIBLE_MODEL` and `COMPATIBLE_BASE_URL` are both required. If you run the agent service in Docker while vLLM runs on the host, use
`COMPATIBLE_BASE_URL=http://host.docker.internal:8000/v1` instead of `localhost`.

The model then appears in the service and Streamlit app as `openai-compatible` and is selected as the default.

