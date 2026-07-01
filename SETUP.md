# CROPI — cloud VM setup (defaults for 2× RTX 4090, adapts to any GPU count)

End-to-end setup for running the CROPI select→RL loop on a plain GPU VM.
Defaults target a 2× RTX 4090 (24GB) box; §4 shows the one-liner to adapt to any GPU count. This is the
fork's scripted layer (`scripts/setup_env.sh` + `scripts/install.sh`) on top of the
upstream pipeline (`cropi/scripts/run_cropi.sh`). All heavy artefacts — uv venvs,
models, data, HF cache, checkpoints — live under one workspace (`$CROPI_WORK`) so the
home disk stays small.

> These scripts mirror our weasel/tads conventions: source `scripts/setup_env.sh`
> once per shell, then run stage scripts. `setup_env.sh` only **warns** about missing
> paths — it never aborts your shell.

CROPI needs **two isolated environments** (their deps conflict):

| env | manager | holds | used by |
|---|---|---|---|
| **cropi** | `uv` venv | torch 2.4.0 cu124 + `traker`/`fast_jl` + `transformers<5` | scoring & selection (`cropi-get-grad`, `cropi-compute-inf-score`, `cropi-select`) |
| **verl** | separate venv | `verl` + `vllm` (+ flash-attn, ray) | RL (GRPO) training and rollout — pointed to by `RL_PYTHON` |

## 0. Clone + env

The fork already tracks both remotes (`origin` = your fork, `upstream` = thu-coai/CROPI):

```bash
git clone git@github.com:33modeling/CROPI.git
cd CROPI
# On a plain cloud VM there is no /group-volume — point the workspace at a large disk:
export GROUP_VOLUME=/mnt/data          # or: export CROPI_WORK=/workspace/cropi
echo 'export GROUP_VOLUME=/mnt/data' >> ~/.bashrc   # so new shells / tmux inherit it
source scripts/setup_env.sh            # derives venvs/models/data/cache under $CROPI_WORK
```

## 1. Install (once, needs internet)

```bash
sudo apt-get update && sudo apt-get install -y git build-essential tmux   # fresh image
nvidia-smi                             # confirm the driver sees your GPUs (and note how many)
bash scripts/install.sh all            # bootstraps uv, then builds the cropi + verl envs
```
- **cropi** is installed exactly per the upstream README (torch 2.4.0 cu124, `-e .`,
  `traker`, `fast_jl`). `fast_jl` compiles a CUDA extension — it needs a **CUDA 12.x
  toolkit (`nvcc`)** on PATH. Most GPU images ship it; if the build fails, install the
  toolkit and re-run `bash scripts/install.sh cropi`.
- **verl** is the version-sensitive part (torch/vLLM/flash-attn are coupled). `install.sh
  verl` does a best-effort `uv pip install verl`; pin it to your CUDA with
  `VERL_PIP_SPEC='verl==<x.y.z>'` per the
  [verl install guide](https://verl.readthedocs.io/en/latest/start/install.html).
  If you already have a working verl env elsewhere, skip this and just
  `export RL_PYTHON=/path/to/verl-env/bin/python`.

## 2. Base model

```bash
source scripts/setup_env.sh
cropi_activate
huggingface-cli download "$HFID_BASE_MODEL" --local-dir "$BASE_MODEL_PATH"   # Qwen2.5-1.5B-Instruct
```
(Reuse an existing local copy by pointing `BASE_MODEL_PATH` at it before sourcing.)

## 3. Data layout — what CROPI needs (NOT included)

CROPI selects over **pre-collected rollouts and their projected gradients**; the repo
ships no data. `run_cropi.sh` expects exactly this tree under `$DATA_ROOT`:

```text
$DATA_ROOT/
  <train_dataset>/                 # e.g. gsm_math_dsr_test
    train_qwen.parquet             # raw train prompt pool
    <model_name>/                  # e.g. Qwen2.5-1.5B-Instruct_curriculum
      train_<infer_note>.jsonl              # rollouts: prompt/answer/responses/rewards
      train_<infer_note>_grad_<proj_note>.jsonl.<rank>   # projected gradient shards
  <valid_dataset>/                 # e.g. gsm8k, math
    valid_qwen.parquet
    <model_name>/
      valid_<valid_infer_note>.jsonl
      valid_<valid_infer_note>_grad_<proj_note>.jsonl.<rank>
```

The `<infer_note>` / `<proj_note>` strings are built by `run_cropi.sh` from
`PROMPT_TYPE`, `TEMPERATURE`, `N_SAMPLES`, `SEED`, `PROJECTION_METHOD`, `PROJ_DIM`,
`SPARSE_DIM` — keep those consistent between generation and selection.

**Generating rollouts (optional, external tooling).** `cropi/inference/infer_trainset.sh`
and `infer_validset.sh` wrap the Qwen2.5-Math evaluation script `math_eval_save_logprob.py`
(vLLM-based). Point `MATH_EVAL_ENTRYPOINT` at your copy of that script
([Qwen2.5-Math eval](https://github.com/QwenLM/Qwen2.5-Math)) and run them in the **verl**
env (it has vLLM). Gradient shards are then produced by `cropi-get-grad`, which
`run_cropi.sh full` invokes automatically between rounds.

## 4. Run

Always pass `$DATA_ROOT` explicitly (it is a positional arg to `run_cropi.sh`).
`DRY_RUN=1` prints the whole command chain without executing — use it first.

```bash
source scripts/setup_env.sh && cropi_activate

# (a) one selection stage — no GPU-heavy RL, just scoring + select
bash cropi/scripts/run_cropi.sh select-only "$DATA_ROOT" Qwen2.5-1.5B-Instruct_curriculum

# (b) full iterative loop: select -> RL -> recompute grads -> select -> RL ...
BASE_MODEL_PATH="$BASE_MODEL_PATH" RL_PYTHON="$RL_PYTHON" \
TRAIN_DATA_NAMES=gsm_math_dsr_test \
VALID_DATA_NAMES=gsm8k,math \
NUM_RL_ROUNDS=2 \
bash cropi/scripts/run_cropi.sh full "$DATA_ROOT" Qwen2.5-1.5B-Instruct_curriculum
```

**Different GPU count?** The defaults assume 2 GPUs. On an `N`-GPU machine set both the
RL world size and the gradient-shard fan-out to `N` before running (they must match — see
the `NUM_PARALLEL` note below):

```bash
export RL_NUM_GPUS=N NUM_PARALLEL=N     # e.g. N=4 or N=8
# larger VRAM (A100 80GB) can also raise the micro-batches back toward the paper values:
export RL_PPO_MICRO_BATCH_SIZE_PER_GPU=16 RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=16
# a >1.5B base that won't fit one GPU also needs RL_TP_SIZE>1 (vLLM tensor-parallel)
```

`setup_env.sh` already exports the 2×4090 defaults, so on a 2-GPU box you don't repeat them:

| knob | 2×4090 default | paper (8×A100) | why |
|---|---|---|---|
| `RL_NUM_GPUS` | `2` | 8 | FSDP world size for RL |
| `RL_TP_SIZE` | `1` | 2 | 1.5B fits on one 4090 → no vLLM tensor-split |
| `NUM_PARALLEL` | `2` | 8 | **must equal visible GPU count** — `cropi-get-grad` pins shard *k* to `gpu = k % NUM_PARALLEL` |
| `RL_PPO_MICRO_BATCH_SIZE_PER_GPU` | `4` | 16 | fit 24GB; raise if VRAM allows |
| `RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU` | `4` | 16 | same |
| `RL_GPU_MEMORY_UTILIZATION` | `0.6` | 0.6 | vLLM rollout headroom |

The micro-batch defaults are conservative **starting points** — watch `nvidia-smi` on
the first steps and raise them until you approach the VRAM ceiling. If RL OOMs, also try
`actor_rollout_ref.actor.fsdp_config.param_offload=True` (edit the flag block in
`cropi/scripts/run_cropi.sh` or lower `RL_MAX_RESPONSE_LENGTH`).

**Keep it alive across SSH drops** — RL rounds are long. Run in `tmux`:
```bash
tmux new -s cropi 'source scripts/setup_env.sh && cropi_activate && \
  BASE_MODEL_PATH="$BASE_MODEL_PATH" RL_PYTHON="$RL_PYTHON" \
  bash cropi/scripts/run_cropi.sh full "$DATA_ROOT" Qwen2.5-1.5B-Instruct_curriculum'
# detach: Ctrl-b d
```

## 5. Workspace layout
```
$CROPI_WORK/                       ← everything heavy (default $GROUP_VOLUME/$USER/cropi)
├── venvs/{cropi,verl}/            ← the 2 uv envs
├── models/Qwen2.5-1.5B-Instruct/  ← BASE_MODEL_PATH
├── data/<dataset>/<model>/...     ← DATA_ROOT (parquet + rollout jsonl + grad shards)
├── checkpoints/cropi_rl/iter*/    ← CKPT_ROOT (verl actor + exported huggingface/)
└── cache/{huggingface,uv}
~/CROPI/                           ← this code checkout (small; home disk)
```
After each RL round the actor is exported to `…/iter<i>/global_step_<N>/actor/huggingface/`
so the next CROPI round reuses it directly.

## 6. Notes / troubleshooting

- **`fast_jl` won't build** → no CUDA toolkit. Install CUDA 12.x (`nvcc`) and re-run the
  cropi install. It projects gradients on-GPU; there is no pure-CPU fallback.
- **`NUM_PARALLEL` > GPU count** → `cropi-get-grad` shards target `cuda:2..` that don't
  exist and hang/crash. Keep `NUM_PARALLEL == RL_NUM_GPUS == 2` here.
- **verl / vLLM version drift** is the usual failure point (mirrors weasel's AgentLab
  caveat). The **cropi** side (scoring/selection) is version-stable; only the **verl**
  env is sensitive. Pin via `VERL_PIP_SPEC` and follow the verl install guide.
- **Disk** — one base model + 2 RL rounds + venvs + HF cache fit comfortably in ~80GB;
  budget more if you keep every `iter*/global_step_*` checkpoint. On ephemeral instance
  storage, copy `checkpoints/**/huggingface/` and selected parquet off `$CROPI_WORK`
  before terminating.
- **wandb** — `RL_USE_WANDB=1` (needs `WANDB_API_KEY`); default is console-only.
