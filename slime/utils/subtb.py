"""NTP SubTB on a finite-horizon prefix tree with a fixed reference policy.

The scalar head predicts g(s) = log F(s) - log p_ref(prefix | prompt).
At a terminal state g(x) = reward(x) / alpha. This change of coordinates
turns action log probabilities into log(pi / pi_ref), without changing SubTB.
"""

import math

import torch


def mean_engine_logprob_gap(engine_log_probs, actor_log_probs, loss_masks):
    """Mean (rollout-engine log-prob - actor log-prob) over masked response tokens.

    The engines report the log-probs they sampled with; the actor recomputes the same
    tokens with its current weights. A healthy round agrees to ~1e-3 nats. A systemic
    gap means the engines are not running the actor's weights: ``--colocate`` releases
    the engines' memory every round and, with ``--sglang-enable-weights-cpu-backup
    False``, ``resume_memory_occupation`` hands the weights back empty, so only
    ``update_weights`` refills them. When that sync is missing the engines sample
    uniform noise whose per-token log-prob is exactly ``-log(vocab_size)`` (-11.93 for
    Qwen3), which is what a degenerate "repetition soup" round looks like in the
    metrics: job 114473 reported -11.93 against the actor's -13.20.

    Returns None when the comparison is not available (no engine log-probs, or nothing
    survives the loss mask).
    """
    if not engine_log_probs or not actor_log_probs or not loss_masks:
        return None

    delta_sum = 0.0
    weight_sum = 0.0
    for engine_sample, actor_sample, mask in zip(engine_log_probs, actor_log_probs, loss_masks):
        mask = mask.flatten().to(torch.float32)
        engine_sample = engine_sample.flatten().to(torch.float32)
        actor_sample = actor_sample.flatten().to(torch.float32)
        width = min(mask.numel(), engine_sample.numel(), actor_sample.numel())
        if width == 0:
            continue
        delta_sum += float(((engine_sample[:width] - actor_sample[:width]) * mask[:width]).sum())
        weight_sum += float(mask[:width].sum())
    if weight_sum == 0:
        return None
    return delta_sum / weight_sum


def add_subtb_arguments(parser):
    parser.add_argument("--subtb-flow-init", choices=("zero", "random"), default="zero")
    parser.add_argument("--subtb-alpha", type=float, default=1.0)
    parser.add_argument("--subtb-num-spans", type=int, default=64,
                        help="Legacy sampler only: extra random spans.")
    parser.add_argument("--subtb-sampling", choices=("window", "legacy"), default="window")
    parser.add_argument("--subtb-window-size", type=int, default=64)
    parser.add_argument("--subtb-num-windows", type=int, default=4,
                        help="Random windows, in addition to one terminal window.")
    parser.add_argument("--subtb-length-lambda", type=float, default=1.0)
    parser.add_argument("--subtb-full-weight", type=float, default=0.1)
    parser.add_argument("--subtb-seed", type=int, default=1234)
    parser.add_argument(
        "--subtb-flow-inner-steps",
        type=int,
        default=1,
        help=(
            "Flow optimizer steps per rollout. All steps reuse the actor's cached "
            "log-probs/rewards of that rollout, which keeps the fast (flow) variable "
            "ahead of the slow (actor) one. The actor still takes one step per rollout."
        ),
    )
    parser.add_argument(
        "--subtb-flow-warmup-steps",
        type=int,
        default=0,
        help=(
            "Leading rollouts that only fit the flow: the actor still runs the "
            "reference/actor forwards that publish log-probs and still re-synchronises "
            "its (unchanged) weights to the rollout engines, but takes no optimizer "
            "step. The sync is not optional: under --colocate every round releases and "
            "resumes the engines' memory without a CPU weight backup, so the engines "
            "hold empty weights until the next update_weights."
        ),
    )
    parser.add_argument(
        "--subtb-engine-gap-abort",
        type=float,
        default=0.0,
        help=(
            "Safety rail: abort the run when the mean rollout-engine log-prob differs "
            "from the actor's recomputed log-prob by more than this many nats in a "
            "round. Healthy rounds sit at ~1e-3; an engine that lost its weights after "
            "release_memory_occupation samples uniform noise and reports ~1 nat or "
            "more (every token at -log(vocab_size)). 0 disables the check."
        ),
    )
    return parser


def validate_subtb_args(args):
    if args.loss_type != "subtb_loss":
        return
    if not math.isfinite(args.subtb_alpha) or args.subtb_alpha <= 0:
        raise ValueError("SubTB alpha must be finite and positive")
    if args.subtb_num_spans < 0:
        raise ValueError("SubTB span count must be nonnegative")
    validate_window_config(args.subtb_window_size, args.subtb_num_windows,
                           args.subtb_length_lambda, args.subtb_full_weight)
    required = {
        "train_backend": "megatron", "context_parallel_size": 1,
        "pipeline_model_parallel_size": 1, "num_critic_only_steps": 0,
        "rollout_temperature": 1.0, "attention_dropout": 0.0, "hidden_dropout": 0.0,
        "partial_rollout": False, "use_rollout_logprobs": False,
        "use_tis": False, "use_kl_loss": False, "kl_coef": 0.0,
        "calculate_per_token_loss": False, "critic_train_only": False,
        "critic_token_loss": False, "keep_old_actor": False,
        "use_dynamic_global_batch_size": False,
        "enable_mtp_training": False,
    }
    for name, expected in required.items():
        if getattr(args, name, expected) != expected:
            raise ValueError(f"NTP SubTB currently requires {name}={expected!r}")
    if args.global_batch_size != args.rollout_batch_size * args.n_samples_per_prompt:
        raise ValueError("SubTB requires one optimizer step per rollout for synchronized joint gradients")
    if not args.enable_weights_backuper or not args.compute_advantages_and_returns:
        raise ValueError("SubTB requires weight backups and the pre-training forward/sync path")
    if args.ref_update_interval is not None or not args.ref_load:
        raise ValueError("SubTB requires a fixed --ref-load and no reference updates")
    if getattr(args, "use_opd", False):
        raise ValueError("SubTB does not support OPD")
    if args.actor_num_nodes != args.critic_num_nodes or args.actor_num_gpus_per_node != args.critic_num_gpus_per_node:
        raise ValueError("SubTB requires matching actor/flow parallel topology")
    if args.subtb_flow_inner_steps < 1:
        raise ValueError("SubTB flow inner steps must be positive")
    if args.subtb_flow_warmup_steps < 0:
        raise ValueError("SubTB flow warmup steps must be nonnegative")
    if args.subtb_flow_warmup_steps:
        if not getattr(args, "use_critic", True):
            raise ValueError("SubTB flow warmup requires the flow model (critic role)")
        if getattr(args, "num_rollout", 0) <= args.subtb_flow_warmup_steps:
            raise ValueError("SubTB flow warmup must leave at least one joint rollout")
    if args.subtb_flow_warmup_steps < 0:
        raise ValueError("SubTB flow warmup steps must be nonnegative")


def subtb_flow_warmup_active(args, rollout_id: int) -> bool:
    """True while the actor is frozen so that the flow can be fitted first.

    The actor must still run its forwards during warmup: the flow's residual needs
    the actor's log-probs and the reference log-probs. Only the actor's optimizer
    step is skipped; its (unchanged) weights are still re-synchronised to the
    rollout engines, which ``--colocate`` requires after every release/resume.
    """
    if getattr(args, "loss_type", None) not in ("subtb_loss", "subtb_flow_loss"):
        return False
    warmup_steps = getattr(args, "subtb_flow_warmup_steps", 0) or 0
    return warmup_steps > 0 and rollout_id < warmup_steps


def validate_subtb_sample(sample, eos_id, horizon):
    """Enforce the bounded prefix-tree terminal convention before training."""
    status = sample.status.name
    if status not in ("COMPLETED", "TRUNCATED"):
        raise ValueError("SubTB cannot train on aborted/incomplete rollouts")
    if not 1 <= sample.response_length <= horizon:
        raise ValueError("SubTB response must contain between 1 and H actions")
    if status == "TRUNCATED" and sample.response_length != horizon:
        raise ValueError("SubTB truncation must be the configured action horizon")
    if status == "COMPLETED" and sample.tokens[-1] != eos_id:
        raise ValueError("SubTB completion must include EOS, not a custom stop string")


def subtb_spans(length, num_spans=64, seed=1234, device=None):
    """Deterministic per-trajectory spans shared by the two model workers.

    Include all edges and the full path; sample additional intervals uniformly
    from all endpoint pairs (duplicates are retained as sampling multiplicity).
    O(T + num_spans) memory, never a T-by-T matrix.
    """
    if length < 1:
        raise ValueError("SubTB needs a nonempty response")
    starts = torch.arange(length)
    ends = starts + 1
    starts = torch.cat((starts, torch.tensor([0])))
    ends = torch.cat((ends, torch.tensor([length])))
    if num_spans:
        generator = torch.Generator().manual_seed(int(seed))
        first = torch.randint(length + 1, (num_spans,), generator=generator)
        other = torch.randint(length, (num_spans,), generator=generator)
        other += (other >= first).long()
        starts = torch.cat((starts, torch.minimum(first, other)))
        ends = torch.cat((ends, torch.maximum(first, other)))
    return starts.to(device), ends.to(device)


def validate_window_config(window_size, num_windows, length_lambda, full_weight):
    if window_size < 1 or num_windows < 0:
        raise ValueError("SubTB window size must be positive and random window count nonnegative")
    if not math.isfinite(length_lambda) or length_lambda <= 0:
        raise ValueError("SubTB length lambda must be finite and positive")
    if not math.isfinite(full_weight) or not 0 <= full_weight <= 1:
        raise ValueError("SubTB full trajectory weight must be in [0, 1]")


def subtb_windows(length, window_size=64, num_windows=4, seed=1234):
    """Uniform starts sampled with replacement, plus a mandatory terminal window.

    Windows contain actions [start, end), hence end-start+1 boundary states.
    For a short response use the complete response once. Random windows may
    overlap or duplicate the terminal window; each occurrence gets equal weight.
    """
    if length < 1 or window_size < 1 or num_windows < 0:
        raise ValueError("Invalid SubTB trajectory/window lengths or count")
    if length <= window_size:
        return [(0, length)]
    generator = torch.Generator().manual_seed(int(seed))
    starts = torch.randint(length - window_size + 1, (num_windows,), generator=generator).tolist()
    return [(start, start + window_size) for start in starts] + [(length - window_size, length)]


def subtb_loss(log_probs, ref_log_probs, flows, reward, *, alpha=1.0, num_spans=64, seed=1234,
               sampling="window", window_size=64, num_windows=4, length_lambda=1.0,
               full_weight=0.1, return_components=False):
    """Window-local weighted SubTB plus an explicitly weighted complete path.

    flows[t] predicts g(s_t) BEFORE action t; only the actual trajectory terminal
    is replaced by reward/alpha. The peer model and reference are detached by
    the caller. Legacy sampling remains available to reproduce the first smoke.
    """
    if log_probs.ndim != 1 or log_probs.shape != ref_log_probs.shape or flows.shape != log_probs.shape:
        raise ValueError("Expected aligned 1D action probabilities and pre-action flows")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    validate_window_config(window_size, num_windows, length_lambda, full_weight)
    reward = torch.as_tensor(reward, dtype=flows.dtype, device=flows.device).detach()
    terminal = (reward / alpha).reshape(1)
    states = torch.cat((flows, terminal))
    prefix = torch.cat((log_probs.new_zeros(1), (log_probs - ref_log_probs.detach()).cumsum(0)))
    full_loss = (states[0] + prefix[-1] - states[-1]).square()
    if sampling == "legacy":
        i, j = subtb_spans(len(log_probs), num_spans, seed, log_probs.device)
        loss = (states[i] + prefix[j] - prefix[i] - states[j]).square().mean()
        components = {"legacy_loss": loss, "full_loss": full_loss}
    elif sampling == "window":
        windows = subtb_windows(len(log_probs), window_size, num_windows, seed)
        width = windows[0][1] - windows[0][0]
        i, j = torch.triu_indices(width + 1, width + 1, offset=1, device=log_probs.device)
        # Stable normalized lambda**length, even for large or small lambda.
        weights = ((j - i).to(log_probs.dtype) * math.log(length_lambda)).softmax(0)
        starts = torch.tensor([start for start, _ in windows], device=log_probs.device)[:, None]
        i, j = i[None, :] + starts, j[None, :] + starts
        residual = states[i] + prefix[j] - prefix[i] - states[j]
        window_loss = (residual.square() * weights).sum(-1).mean()
        loss = (1 - full_weight) * window_loss + full_weight * full_loss
        components = {"window_loss": window_loss, "full_loss": full_loss}
    else:
        raise ValueError(f"Unknown SubTB sampler: {sampling}")
    if return_components:
        return loss, {key: value.detach() for key, value in components.items()}
    return loss
