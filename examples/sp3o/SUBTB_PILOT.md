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
minimum requirement. Actor and flow each use TP4, PP1, CP1, sequence parallel,
BF16, optimizer CPU offload and activation recomputation. SGLang engine TP is2.
Actor LR is1e-6, flow LR4e-6, Adam betas(.9,.98), weight decay.1.

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
export NUM_STEPS_PER_ROLLOUT=1 NUM_ROLLOUT=100 SAVE_INTERVAL=20 EVAL_INTERVAL=20
export EVAL_MAX_RESPONSE_LEN=8192 CUDA_GRAPH_MAX_BATCH_SIZE=8 SGLANG_MEM_FRACTION_STATIC=0.65
export SUBTB_FLOW_INIT=zero SUBTB_SAMPLING=window SUBTB_ALPHA=1
export SUBTB_WINDOW_SIZE=64 SUBTB_NUM_WINDOWS=4 SUBTB_LENGTH_LAMBDA=1 SUBTB_FULL_WEIGHT=0.1
export WANDB_MODE=online WANDB_PROJECT=SP3O-SubTB USE_WANDB=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Use an existing W&B login. Select unused Ray ports if sharing a node.
bash examples/sp3o/train_subtb.sh \
  --sglang-disable-cuda-graph --sglang-attention-backend triton \
  --sglang-context-length 9216 --sglang-max-running-requests 16 \
  --rollout-max-prompt-len 1024 --rollout-top-p 1.0 --eval-top-p 1.0 \
  --n-samples-per-eval-prompt 4 \
  --save-debug-rollout-data "$OUTPUT_DIR/$EXPERIMENT_NAME/samples/{rollout_id}.pt"
```

This launcher starts Ray by default. With an existing allocated Ray cluster,
set START_RAY=0, RAY_ADDRESS and RAY_DASHBOARD_PORT appropriately. Do not
terminate another job's Ray processes. Use a new experiment name/output path.
For a capacity run use NUM_ROLLOUT=3, SAVE_INTERVAL=3, EVAL_INTERVAL=3 and
capacity.yaml; restart the efficacy run from base weights after it passes.

## Environment and evidence

The validated environment uses Python3.12, torch2.9.1+cu128,
transformers4.57.1, numpy1.26.4, Transformer Engine2.10 and FlashAttention2.7.4.post1.
Megatron-LM revision3714d81d418c9f1bca4594fc35f9e8289f652862 and SGLang
revisionbbe9c7eeb520b0a67e92d133dfc137a3688dc7f2 have the project's Docker
patches applied; follow the repository environment setup rather than using
unpatched arbitrary upstream versions. CUDA extensions must match the new GPU.

75 SubTB/SP3O/critic-head tests passed on CPU (job113634). TP2 short execution
passed (113635). TP4 capacity (113639) passed three512-response joint updates,
real positive rewards,8192-token horizon coverage, evaluations and checkpoints.
The 100-round job113640 was submitted; no final efficacy result is claimed.

Primary metric is eval/dapo_pilot (average correctness, not pass@4).
rollout/raw_reward tracks training correctness; rollout/values is mean g,
not PPO value or a calibrated correctness probability. Window/full losses
measure consistency, not task accuracy. W&B includes per-GPU system metrics.
