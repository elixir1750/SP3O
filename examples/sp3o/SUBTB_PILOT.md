# Reproducing the 4B SubTB pilot

Run these commands inside a compute allocation, with the project's Megatron /
SGLang environment installed. Configure paths for your server; `local/lumia/`
is site-local and is intentionally not distributed. No credentials are bundled.

## Data

Download `zhuzilin/dapo-math-17k` at revision
`2e65612930298bde4c5d58fd97b3f23a483aaff9`. The source JSONL SHA256 is
`cc9c39c2aa19177abe9464741e121cf4cac90fd25484ef3cdf86535101e3a5b6`.
It has 17,398 records, each occurring once; it is not the million-row repeated
original release. Preparation collapses normalized identical prompts, excludes
entire conflicting-label groups, and filters prompts over 1,024 tokenizer
tokens. Seed 1234 produces 16,919 training questions and 256 held-out questions.
There are 205 duplicate rows collapsed, 10 conflicting rows excluded (5 groups),
and 8 overlong prompts excluded. No semantic/paraphrase deduplication is claimed.

```bash
python examples/sp3o/prepare_subtb_pilot.py \
  --source /path/to/dapo-math-17k.jsonl \
  --model /path/to/Qwen3-4B-Base \
  --output /path/to/new-pilot-data
```

The new directory contains train/eval JSONL, capacity/effect evaluation YAML,
a manifest and the excluded conflicting-label groups. The source is unchanged.

## Training

Request 8 GPUs on one node. The LUMIA capacity run passed with 8 RTX6000 Ada
48GB GPUs, 64 CPUs and 640GB host RAM. This is a tested allocation, not a
minimum requirement. Actor and flow each use TP2 (tensor parallelism 2, as in the
upstream example) with DP2, PP1, CP1, sequence parallel, BF16, optimizer CPU
offload and activation recomputation. SGLang engine TP is2. Actor LR is1e-6, flow
LR3e-5, Adam betas(.9,.98), weight decay.1. The schedule is two-timescale: the
first `SUBTB_FLOW_WARMUP_STEPS` rollouts only fit the flow with the actor frozen,
then every rollout takes one actor step and `SUBTB_FLOW_INNER_STEPS` flow steps.
TP2 x DP2 measured 47s per flow step against 10.8 min at TP4 x DP1 (job 113925),
because the TP4/DP1 layout was activation-memory bound rather than FLOP bound.

```bash
export HF_CHECKPOINT=/path/to/Qwen3-4B-Base
export MEGATRON_CHECKPOINT="$HF_CHECKPOINT"
export MEGATRON_LM=/path/to/Megatron-LM
export PYTHONPATH="$MEGATRON_LM:$PWD"
export PROMPT_DATA=/path/to/new-pilot-data/train.jsonl
export EVAL_CONFIG=/path/to/new-pilot-data/effect.yaml
export OUTPUT_DIR=/path/to/experiments
export EXPERIMENT_NAME=subtb-4B-pilot-seed1234
export MODEL_SIZE=4B ACTOR_GPUS=4 TOTAL_GPUS=8 ROLLOUT_GPUS_PER_ENGINE=2
export SEED=1234 SEQ_LENGTH=9216 MAX_RESPONSE_LEN=8192 MAX_TOKENS_PER_GPU=9216
export ROLLOUT_BATCH_SIZE=64 N_SAMPLES_PER_PROMPT=8 OVER_SAMPLING_BATCH_SIZE=64
export SUBTB_FLOW_WARMUP_STEPS=12 SUBTB_FLOW_INNER_STEPS=2
export SUBTB_FLOW_WARMUP_TARGET_GAP=0.05 SUBTB_FLOW_WARMUP_MIN_STEPS=4
export CRITIC_LR=3e-5 TP_SIZE=2
export NUM_STEPS_PER_ROLLOUT=1 NUM_ROLLOUT=112 SAVE_INTERVAL=20 EVAL_INTERVAL=20
export EVAL_MAX_RESPONSE_LEN=8192 CUDA_GRAPH_MAX_BATCH_SIZE=256 SGLANG_MEM_FRACTION_STATIC=0.7
export SUBTB_FLOW_INIT=zero SUBTB_SAMPLING=window SUBTB_ALPHA=1
export SUBTB_WINDOW_SIZE=64 SUBTB_NUM_WINDOWS=4 SUBTB_LENGTH_LAMBDA=1 SUBTB_FULL_WEIGHT=0.1
export WANDB_MODE=online WANDB_PROJECT=SP3O-SubTB USE_WANDB=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Use an existing W&B login. Select unused Ray ports if sharing a node.
bash examples/sp3o/train_subtb.sh \
  --sglang-disable-cuda-graph --sglang-attention-backend triton \
  --sglang-context-length 9216 \
  --rollout-max-prompt-len 1024 --rollout-top-p 1.0 --eval-top-p 1.0 \
  --n-samples-per-eval-prompt 4 \
  --save-debug-rollout-data "$OUTPUT_DIR/$EXPERIMENT_NAME/samples/{rollout_id}.pt"
```

The upstream Qwen3-4B example passes neither an engine cap nor a client
concurrency, which is unsafe at this geometry (512 responses per round, ~8k token
tail): with no cap the client may hold 512 requests per engine, which drove the
~390k token KV pool to 100% and killed a 100-round job with router 503s (job
113811). The local launcher therefore pins both to the same value,
`--sglang-max-running-requests 32 --sglang-server-concurrency 32`. The cap
should still be sized for the worst *plausible* round rather than the average,
but note that the 8002-of-8192 round that originally motivated 32 (job 113925)
was not a long-answer round at all: it was the empty-engine bug described below,
where every token was uniform noise. With clean engines the curated data
truncates about 1% of responses (screening 114477), so 32 (256k tokens, ~66% of
the ~390k pool) is the conservative point on the measured curve: cap16 142,
cap32 216, cap64 235, cap96 249 tokens/gpu/s, all with zero 503s, i.e. 32 costs
~13% against 96. Retune it only with a bounded sweep.

Warmup rounds must re-push the actor's weights. Under `--colocate` each round
releases the engines' memory and, because the launcher runs with
`--sglang-enable-weights-cpu-backup=False`, `resume_memory_occupation` does not
bring the weights back: `update_weights` is what refills them. Skipping it in
warmup (an earlier revision of `train.py` did) makes the engines sample uniform
noise, which shows up as exactly `-log(151936) = -11.93` per token against the
actor's recomputed `-13.2`, 94% truncation and reward 0 for 45 minutes per round
(job 114473; the earlier "degenerate rounds" were the same bug). The launcher now
passes `--subtb-engine-gap-abort 0.1` and every round logs
`rollout/engine_logprob_gap`, so the driver stops after one bad round instead of
training on noise.

Degenerate groups are dropped before training
(`--dynamic-sampling-filter-path slime.utils.subtb_filter.subtb_group_filter`): every
response truncated, or no response parsable. Dumps show 0% of groups are dropped
in normal rounds and 100% in a degenerate one, so the rule is free when the policy
behaves and protective when it does not; a bounded drop budget keeps a bad region
from starving the rollout. Dropping zero-variance groups instead would cost ~50%
extra rollout in every round and is off by default (`SUBTB_FILTER_ZERO_STD=1`).
Drop counts are logged as `rollout/dynamic_filter/drop_<reason>`; the held-out
evaluation is unaffected.

This launcher starts Ray by default. With an existing allocated Ray cluster,
set START_RAY=0, RAY_ADDRESS and RAY_DASHBOARD_PORT appropriately. Do not
terminate another job's Ray processes. Use a new experiment name/output path.
For a capacity run use NUM_ROLLOUT=3, SAVE_INTERVAL=3, EVAL_INTERVAL=3 and
capacity.yaml; restart the efficacy run from base weights after it passes.

## Environment and evidence

## Effect run: the configuration we actually submit

The 100-joint-round run is submitted as the `effect` stage of the local launcher,
which refuses to start unless the capacity/smoke run it points at contains a
`VALIDATED` marker. Its settings, as verified in the smoke:

| Item | Value |
| --- | --- |
| Resources | 1 node x 8 GPU (L40S or ADA6000), 64 CPU, 900 GB host RAM, 3-day limit |
| Data | curated warmup prefix (768 mixed-outcome prompts) followed by the 16,151 remaining DAPO train rows, `ROLLOUT_SHUFFLE=0` so the warmup consumes the prefix in order; evaluation on the fixed 256 held-out questions x 4 |
| Rollout | 64 prompts x 8 responses = 512 per round, 1x over-sampling, temperature 1.0, top-p 1.0, context 9216 (prompt <= 1024, response <= 8192) |
| Engines | 4 x SGLang TP2, `max_running_requests` = client concurrency = 32, mem-fraction 0.7, CUDA graphs disabled, triton attention |
| Training topology | actor TP2 x DP2 (4 GPUs) + flow/critic TP2 x DP2 (4 GPUs), BF16, sequence parallel, recompute 1 layer, optimizer CPU offload |
| Joint-step memory | `--recompute-method uniform --recompute-num-layers 1` is the saving setting (the flag is a chunk size, not a count) plus `--log-probs-chunk-size 1024`: on a 44 GB L40S the unbounded fp32 log-prob workspace for one long sample is 1.8-2.8 GB and the joint train step OOMs by ~0.2 GB (jobs 114577, 114740) |
| Optimizer | Adam (0.9, 0.98), weight decay 0.1, constant LR: actor 1e-6, flow 3e-5 |
| Per round | actor exactly 1 optimizer step, flow K = 2 steps on the same cached snapshot |
| Objective | NTP SubTB, alpha 1, zero-initialised flow head, 4 random 64-action windows + mandatory terminal window, lambda 1, full-path weight 0.1, no PPO clipping/GAE/TIS/partial rollout |
| Two-timescale | exactly 12 warmup rounds with the actor frozen (no convergence gate: under a frozen actor the flow's fixed point is the mean reward, while log Z(q) belongs to the joint solution - see SUBTB.md), then >= 100 joint rounds, one actor step and K = 2 flow steps each |
| Update order | simultaneous within a rollout: the actor's step uses the flow snapshot taken before this rollout's flow steps, the flow leads from the next rollout on (a flow-first ordering would need an extra sync + re-evaluation) |
| Actor backup | `train_actor` must end with `weights_backuper.backup("actor")`; it is what the next round restores when switching back to "actor" and what `update_weights` ships to the engines |
| Data hygiene | drop degenerate groups (all 8 truncated, or all 8 unparseable) with a bounded refill budget; forced accepts are logged as "group filter saturated" |
| Weight-sync rail | every round logs `rollout/engine_logprob_gap` and `--subtb-engine-gap-abort 0.1` aborts the run if the engines stop running the actor's weights (empty engines report exactly `-log(151936) = -11.93` per token, i.e. ~1 nat against the actor) |
| Evidence | W&B online, metrics per round, checkpoints every 20 joint rounds, evaluation every 20 rounds, per-round sample dumps, gate writes `pilot-evidence.json` + `VALIDATED` |

Totals for the run: 112 rollouts (12 warmup bound + 100 joint), 224 flow steps,
at least 100 actor updates, estimated 15-22 hours on one 8-GPU node.

The validated environment uses Python3.12, torch2.9.1+cu128,
transformers4.57.1, numpy1.26.4, Transformer Engine2.10 and FlashAttention2.7.4.post1.
Megatron-LM revision3714d81d418c9f1bca4594fc35f9e8289f652862 and SGLang
revisionbbe9c7eeb520b0a67e92d133dfc137a3688dc7f2 have the project's Docker
patches applied; follow the repository environment setup rather than using
unpatched arbitrary upstream versions. CUDA extensions must match the new GPU.

75 SubTB/SP3O/critic-head tests passed on CPU (job113634). TP2 short execution
passed (113635). TP4 capacity (113639) passed three512-response joint updates,
real positive rewards,8192-token horizon coverage, evaluations and checkpoints.
The first 100-round job113640 failed during initial evaluation with router503
errors before any training update. Client concurrency was512 per engine despite
an engine running-request limit of16. The launch above now bounds client
concurrency to16 to avoid flooding the router. Retry113758 was submitted after
CPU argument preflight113757 passed; the retry is not yet validated end-to-end.
No final efficacy result is claimed.

Primary metric is eval/dapo_pilot (average correctness, not pass@4).
rollout/raw_reward tracks training correctness; rollout/values is mean g,
not PPO value or a calibrated correctness probability. Window/full losses
measure consistency, not task accuracy. W&B includes per-GPU system metrics.
