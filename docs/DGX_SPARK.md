# DGX Spark training guide (GB10, 128GB unified, ~1 PFLOP FP4)

## What the hardware means for this project

- **128GB unified LPDDR5X**: the whole training state (weights + optimizer +
  grads + activations for ≤600M models) fits with room to spare. No FSDP, no
  sharding, no offload gymnastics — single-process training.
- **~273 GB/s bandwidth**: modest vs H100 (~3 TB/s). Training will be
  **bandwidth-bound**, not FLOP-bound. Optimize for arithmetic intensity:
  large batches, flash attention, bf16, torch.compile.
- **Blackwell FP4**: great for *inference* (NVFP4); training stays bf16 mixed
  precision. Use FP8 only if the DGX OS stack supports it stably.
- **ARM CPU + NVMe 4TB**: streaming datasets (HF `streaming=True`) + local
  SSD cache; the knowledge index lives on the same SSD as checkpoints.

## Throughput targets (150m, seq 4096, batch ~1M tokens)

| optimization | expected effect |
|---|---|
| raw eager, no flash | ~8–12k tok/s (bandwidth-starved) |
| + SDPA flash (`F.scaled_dot_product_attention`) | ~20–30k tok/s |
| + `torch.compile(mode="max-autotune")` | ~35–55k tok/s |
| + packing (no padding) + large grad-accum microbatch | ~40–60k tok/s |

At 45k tok/s, 80B tokens ≈ 20 days. Measure with the `tok/s` logger in
`train.py` and tune `grad_accum`/microbatch to the largest step that fits.

## Practical checklist

1. **Use the DGX OS CUDA torch build** (ARM aarch64 + CUDA). Verify:
   `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"`.
   Blackwell GB10 needs sm_121a or newer — install a torch nightly/CUDA 12.8+
   if the stable wheel lacks the arch.
2. **Env**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
   `OMP_NUM_THREADS=16`, `TOKENIZERS_PARALLELISM=false` (see `scripts/train.sh`).
3. **Data streaming**: HF streaming with a 10k shuffle buffer; keep a local
   `HF_HOME` on NVMe. Pre-tokenize a local shard if the network becomes the
   bottleneck (packing makes pre-tokenization trivially parallel).
4. **Unified memory**: oversubscription is graceful (no OOM cliff), but
   paging kills throughput — keep microbatch resident. Watch `tegrastats`-style
   monitors on DGX OS.
5. **Checkpoints**: every 1000 steps (~1B tokens) to NVMe; keep last 3 + best
   ppl. `meta.json` records tokens for WSD resume math.
6. **Eval cadence**: ppl every 500 steps; full `eval.sh` at anneal boundaries
   and after SFT/DPO. Log everything to `runs.md`.
7. **Power/thermals**: 140–240W sustained is fine on a desk; ensure airflow
   for multi-week runs. `nvidia-smi dmon` in a tmux pane.

## Fallback if throughput is low

- Drop to seq 2048 for Stage 1 (2× fewer attention FLOPs), extend to 4k/8k
  only in anneal/SFT — RoPE scaling makes this nearly free.
- Start the floor probe at 150m; only scale to 350m/600m once the recipe is
  validated — a fast small run beats a slow big run for iteration.
