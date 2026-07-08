# CROPI / VERL Local Execution Troubleshooting Log
**Date:** 2026-07-08
**Model:** Qwen3.5-9B (Local, `weasel` directory)
**Goal:** Run CROPI full pipeline (Influence scoring -> RL PPO) to check for OOM on 2x RTX 4090.

## Resolved Issues
1. **vLLM V1 Engine Conflict:**
   - **Error:** `ValueError: Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False.`
   - **Fix:** Set `export VLLM_USE_V1=1` before execution.
2. **Missing setuptools/pkg_resources in UV env:**
   - **Error:** `ModuleNotFoundError: No module named 'pkg_resources'` when importing `verl`.
   - **Fix:** Installed older setuptools: `uv pip install "setuptools<70.0.0"`
3. **Data Path / Batch Size Mismatch:**
   - **Error:** `Train dataloader is empty!` (Only 29 items selected, but batch size was 128).
   - **Fix:** Reduced `RL_TRAIN_BATCH_SIZE=16`, `RL_PPO_MINI_BATCH_SIZE=16`, and micro_batch sizes to 2. Overrode data paths to use existing `Qwen3.5-9B_curriculum` gradients (`n4` sampling).
4. **Flash Attention ABI Conflict:**
   - **Error:** `undefined symbol: _ZN3c105ErrorC...`
   - **Fix:** Avoided source build. Installed pre-built official wheel matching PyTorch 2.4 / CUDA 12.4.
5. **Transformers Version / Vision Module Bug:**
   - **Error:** `verl` expects `AutoModelForVision2Seq` which crashes on some versions, but Qwen3.5 requires `transformers>=4.57`.
   - **Fix:** Upgraded transformers to `5.13.0` to support `Qwen3_5ForConditionalGeneration` and hard-patched `verl/workers/fsdp_workers.py` to bypass vision imports.

## Pending / Blocking Issue (Experiment Stopped)
**FSDP Layer Wrapping Failure**
- **Error:** `Exception: Could not find the transformer layer class to wrap in the model.` (in `verl/utils/fsdp_utils.py:114`)
- **Cause:** Ray's FSDP auto-wrapper relies on predefined layer name patterns (like `LlamaDecoderLayer`). It failed to recognize the internal layer structure class for the new `Qwen3.5-9B` architecture.
- **Next Steps:** Modify `verl/utils/fsdp_utils.py` to explicitly append the Qwen3.5 decoder layer class string (e.g., `Qwen3_5DecoderLayer`) to the FSDP wrapping policy list.

## Conclusion on OOM
The pipeline successfully loaded the model weights into GPU RAM (about 400MB base allocation seen initially) without throwing an immediate OOM error on the 2x 24GB RTX 4090 setup. The crash occurred during the FSDP sharding configuration step, implying that memory capacity was sufficient up to the point of model initialization.
