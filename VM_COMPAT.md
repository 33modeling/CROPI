# CROPI VM Compatibility Guide: A100 and H100

This guide captures the current compatibility assumptions inferred from the
recent fork commits and turns them into repeatable VM setup commands.

The short version: keep **two isolated Python environments** and avoid unpinned
`verl`/`vllm` installs.

| env | purpose | pinned baseline |
|---|---|---|
| `cropi` | scoring, selection, projected gradients | `torch==2.4.0` cu124, `traker==0.3.2`, `fast_jl` built with local `nvcc` |
| `verl` | GRPO RL and vLLM rollout | `torch==2.6.0` cu124 -> `vllm==0.8.5` -> `verl==0.4.1` -> `tensordict==0.6.2` |

Why this matrix:

- `cropi/scripts/run_cropi.sh` calls classic `verl.trainer.main_ppo`.
- Bare latest `verl` can move to code paths that expect a newer vLLM async
  server, while this repo pins vLLM to the 0.8 line.
- `vllm` wheels are tightly coupled to CUDA and PyTorch builds. Install torch
  first, then vLLM, then verl, then `tensordict`.
- `fast_jl` must be compiled for the local GPU architecture: A100 is `8.0`,
  H100 is `9.0`.

References:

- PyTorch previous CUDA wheel commands: https://pytorch.org/get-started/previous-versions/
- vLLM GPU install notes on CUDA/PyTorch binary coupling: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
- verl vLLM 0.8 note for `tensordict==0.6.2`: https://verl.readthedocs.io/en/v0.4.1/README_vllm0.8.html
- NVIDIA compute capability table: https://developer.nvidia.com/cuda/gpus
- NVIDIA CUDA compatibility overview: https://docs.nvidia.com/deploy/cuda-compatibility/index.html

## Profile scripts

Use one of these once per shell:

```bash
# A100 80G cluster / old-driver-safe defaults
source scripts/setup_env_a100.sh 4      # 2, 4, or 8 GPUs

# H100/Hopper VM; auto-detects GPU count, or pass 1/2/4/8
source scripts/setup_env_h100.sh 8
```

Both profile scripts export a shared matrix from `scripts/vm_compat.sh`:

```text
CROPI_VM_PROFILE=a100|h100
CROPI_COMPAT_MATRIX=classic-verl0.4-vllm0.8-cu124
CROPI_TORCH_SPEC=torch==2.4.0
VERL_TORCH_SPEC=torch==2.6.0
VLLM_PIP_SPEC=vllm==0.8.5
VERL_PIP_SPEC=verl==0.4.1
TENSORDICT_SPEC=tensordict==0.6.2
TORCH_CUDA_ARCH_LIST=8.0  # A100
TORCH_CUDA_ARCH_LIST=9.0  # H100
```

Inspect the active matrix without changing anything:

```bash
bash scripts/vm_compat.sh print
```

## A100 runbook

The A100 profile keeps CUDA redist at `12.2.2` by default because the observed
cluster nodes advertise CUDA 12.2 through the driver. That avoids building
`fast_jl` PTX with a newer toolkit than the driver can handle.

```bash
cd CROPI
source scripts/setup_env_a100.sh 4

# Build cropi without failing if the VM has no system nvcc.
bash scripts/install_venv.sh cropi

# Assemble a CUDA toolkit under $CROPI_WORK and rebuild fast_jl for sm_80.
bash scripts/setup_cuda_venv.sh

# Build the RL env with the classic verl/vLLM matrix.
bash scripts/install_venv.sh verl

# Verify paths, GPU count, matrix, imports, and model config.
bash scripts/preflight.sh
```

If a previous run installed latest `verl`/`vllm` or rebuilt torch incorrectly:

```bash
bash scripts/repair_vm_libs.sh a100 verl
# or, to re-pin both envs:
bash scripts/repair_vm_libs.sh a100 all
```

## H100 runbook

The H100 profile uses the same RL matrix but compiles local CUDA extensions for
Hopper (`TORCH_CUDA_ARCH_LIST=9.0`). It keeps a separate default workspace
(`$GROUP_VOLUME/$USER/cropi-h100`) so A100 and H100 `fast_jl` builds do not
overwrite each other.

```bash
cd CROPI
source scripts/setup_env_h100.sh 8

bash scripts/install_venv.sh cropi
bash scripts/setup_cuda_venv.sh
bash scripts/install_venv.sh verl
bash scripts/preflight.sh
```

If the H100 driver advertises CUDA 12.4 or newer, `setup_cuda_venv.sh` uses
CUDA redist `12.4.0`; otherwise it falls back to `12.2.2`. Override when needed:

```bash
export CUDA_REDIST_VER=12.2.2   # conservative
export CUDA_REDIST_VER=12.4.0   # newer-driver H100
export TORCH_CUDA_ARCH_LIST=9.0
bash scripts/setup_cuda_venv.sh
```

Repair a drifted H100 environment:

```bash
bash scripts/repair_vm_libs.sh h100 verl
bash scripts/preflight.sh
```

## Common failure patterns

| symptom | likely cause | action |
|---|---|---|
| `run_headless` / async-server errors in `verl.trainer.main_ppo` | latest `verl` with old vLLM | `bash scripts/repair_vm_libs.sh <profile> verl` |
| `ForkingPickler` import error from vLLM | missing `tensordict==0.6.2` with vLLM 0.8 | repair the `verl` env |
| `NVIDIA driver too old` or torch CUDA unavailable | resolver upgraded torch to a CUDA build the driver cannot run | repair the `verl` env; use a newer-driver node if cu124 still fails |
| `fast_jl` build fails because `CUDA_HOME`/`nvcc` is missing | VM has runtime libraries only, no toolkit | `bash scripts/setup_cuda_venv.sh` |
| `fast_jl` works on A100 but fails on H100 | extension was built for `sm_80` only | source `setup_env_h100.sh`, rebuild with `setup_cuda_venv.sh` |
| `cropi-get-grad` targets missing GPUs | `NUM_PARALLEL` exceeds visible GPUs | source profile with the correct GPU count |

## Overrides

All pins can be overridden before running install/repair scripts:

```bash
export VERL_TORCH_SPEC=torch==2.6.0
export VERL_TORCH_INDEX=https://download.pytorch.org/whl/cu124
export VLLM_PIP_SPEC=vllm==0.8.5
export VERL_PIP_SPEC=verl==0.4.1
export TENSORDICT_SPEC=tensordict==0.6.2
```

Use overrides only when you are intentionally changing the whole matrix. Mixing
one newer package into this stack is the common source of the VM-only failures.
