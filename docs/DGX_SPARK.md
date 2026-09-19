# DGX Spark training guide (GB10, 128GB unified, ~1 PFLOP FP4)

## What the hardware means for this project

- **128GB unified LPDDR5X**: CPU and GPU share the memory budget. Small
  parameter counts do not guarantee a training batch fits: activations,
  attention workspaces and vocabulary logits can dominate memory usage.
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
4. **Unified memory**: leave headroom for the OS and dataset buffers. Memory
   exhaustion can cause CUDA allocation errors or an OS OOM kill. Monitor
   system memory with `free -h` alongside the trainer's CUDA peak statistics.
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

## Recovering a 150m OOM

The previous configuration used 64 sequences per microbatch at 4096 tokens
in both stages. A single FP32 vocabulary-logit tensor at that size is about
31 GiB, before loss buffers and activations across 30 layers. The sliding
attention mask also means SDPA backend selection must be measured; calling
SDPA alone does not guarantee Flash Attention is used.

The 150m configuration now starts with 2 sequences per microbatch, activation
checkpointing enabled, and compilation disabled. Pretraining accumulates 128
microbatches and annealing 64, preserving their effective token batches and LR
schedules. This is a conservative starting point, not a measured Spark capacity
or throughput guarantee. Checkpointing trades extra computation for memory.

Launch normally:

```bash
bash scripts/train.sh configs/150m.yaml pretrain
```

The startup log reports the microbatch settings. Step 1 and subsequent logging
intervals report peak allocated and reserved CUDA memory (not total system
memory). If there is ample headroom, decrease `grad_accum` by a factor of two
to double the microbatch; measure again before enabling `compile`.

If the process prints only `Killed`, inspect `sudo dmesg -T | tail -n 80` for
an OOM-killer entry and check `free -h`. Preserve the last training log lines
and the PyTorch/CUDA versions when diagnosing the run.
