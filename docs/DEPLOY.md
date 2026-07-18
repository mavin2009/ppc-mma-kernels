# Deploying Bonsai on Power10 / Power11

End-to-end path from a bare ppc64le machine to serving a PrismML Bonsai
model with the MMA-accelerated 1-bit/ternary kernels.

## 1. Prerequisites

- Power10 or Power11, ppc64le Linux (tested toolchain: GCC 14; GCC >= 10.2
  required for MMA builtins)
- `git`, `cmake` (>= 3.14), `g++`, `make`

RHEL: `dnf install git cmake gcc-c++ make` · Ubuntu: `apt install git cmake g++ make`

## 2. Build

```sh
git clone https://github.com/mavin2009/ppc-mma-kernels
cd ppc-mma-kernels
./scripts/build-bonsai-power.sh
```

The script clones the PrismML llama.cpp fork at the pinned commit the
patch was verified against (`79697f2…`), applies both integration
patches (Q1_0/Q2_0 v3 kernels plus the Q4_K/Q5_K/Q6_K/Q2_K family --
so standard K-quant GGUF models also hit the MMA path), and builds `llama-cli`, `llama-server`,
and `llama-bench` natively with `-mcpu=native`.

Verification status of this integration: the patched fork
**cross-compiles cleanly for ppc64le (GCC 14, `-mcpu=power10`) and the
resulting `llama-cli` executes under `qemu -cpu power10`**; the kernels
themselves are correctness-verified against double-precision references
(see DESIGN.md). End-to-end model inference through the patched
dispatch has not been run in CI (no Power hardware / model weights in
the loop) — do step 5 on first deployment.

## 3. Get a model

Bonsai GGUF weights (Q1_0 1-bit or Q2_0 ternary) and the vision
projector (`mmproj`) are published under the PrismML organization on
Hugging Face (https://huggingface.co/PrismML — pick the GGUF repo for
your model size; exact repo names change as releases move, so browse
rather than trusting a hardcoded name here).

```sh
pip install -U "huggingface_hub[cli]"
hf download <PrismML/model-repo> --include "*q2_0*.gguf" "*mmproj*" --local-dir models/
```

## 4. Run

```sh
cd bonsai-build/llama.cpp
# chat / single prompt
build/bin/llama-cli -m ../../models/<model>.gguf -p "Hello" -n 64

# OpenAI-compatible server with vision
build/bin/llama-server -m ../../models/<model>.gguf \
    --mmproj ../../models/<mmproj>.gguf \
    --host 0.0.0.0 --port 8080
```

Startup prints a system-info line; confirm it shows `MMA = 1`. Thread
count defaults to hardware; on SMT machines `-t <physical cores>`
usually beats the default.

## 5. First-deployment sanity checks

1. **Numerics**: run the same prompt on this build and on an unpatched
   scalar build (`git stash` the patch, rebuild); outputs should match
   token-for-token at temperature 0.
2. **Perf**: `build/bin/llama-bench -m <model>.gguf` — compare pp/tg
   throughput against the unpatched build. Prompt processing should
   improve substantially; token generation (n = 1) improvement is
   smaller by nature (see DESIGN.md future work on a GEMV path).
3. Please report numbers back via a GitHub issue — real Power silicon
   results are the missing piece of this project's benchmarking story.

## 6. Serving as a service (optional)

```ini
# /etc/systemd/system/bonsai.service
[Unit]
Description=Bonsai llama-server (Power MMA)
After=network.target

[Service]
ExecStart=/opt/bonsai/llama.cpp/build/bin/llama-server \
    -m /opt/bonsai/models/model.gguf --mmproj /opt/bonsai/models/mmproj.gguf \
    --host 127.0.0.1 --port 8080
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

- **Patch fails to apply**: the fork moved; check out the pinned commit
  (the build script attempts this) or open an issue.
- **`MMA = 0` in system info**: compiler too old or `-mcpu` didn't
  select power10; ensure GCC >= 10.2 and build natively on the Power
  box (or pass `-mcpu=power10` explicitly).
- **Slow under qemu**: expected; qemu is for correctness only.
