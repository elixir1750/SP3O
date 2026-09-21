import logging
import math

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action
from slime.utils.subtb import add_subtb_arguments, subtb_flow_warmup_active, validate_subtb_args
from slime.utils.sp3o import add_sp3o_arguments, validate_sp3o_args

logger = logging.getLogger(__name__)


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Update primary W&B with SGLang metrics endpoint now that servers are up.
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    if not args.critic_train_only:
        actor_model.update_weights()

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(rollout_id):
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            if args.critic_train_only:
                critic_model.clear_memory()
            else:
                actor_model.clear_memory()

    def save(rollout_id):
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps and not args.critic_train_only):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # SubTB may ramp in with flow-only warmup rounds. The actor process is still
        # started (it must publish log-probs and reference log-probs for the flow) but
        # skips its optimizer step, so its weights are unchanged.
        warmup = subtb_flow_warmup_active(args, rollout_id)

        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        actor_signals = None
        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                actor_signals = ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            actor_signals = ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if actor_signals and getattr(args, "subtb_engine_gap_abort", 0.0) > 0:
            # Cheap weight-sync rail: the engines report their own sampling log-probs,
            # the actor recomputes them with its current weights. A systemic gap means
            # the engines are not running the actor's weights any more (stale copy, or
            # empty weights after resume_memory_occupation) and the rollout is noise.
            engine_gaps = [
                abs(signal["engine_gap"])
                for signal in actor_signals
                if isinstance(signal, dict) and signal.get("engine_gap") is not None
            ]
            if engine_gaps:
                worst_gap = max(engine_gaps)
                logger.info("SubTB engine log-prob gap: %.5f nats (limit %.3f)",
                            worst_gap, args.subtb_engine_gap_abort)
                if worst_gap > args.subtb_engine_gap_abort:
                    raise RuntimeError(
                        f"Rollout engines disagree with the actor by {worst_gap:.4f} nats "
                        f"(limit {args.subtb_engine_gap_abort}). The engines are most likely "
                        "running stale or empty weights, e.g. after release_memory_occupation "
                        "without a following update_weights (colocate mode). Refusing to train "
                        "on these rollouts."
                    )

        if warmup and actor_signals:
            # Diagnostics only, never a gate: while the actor is frozen at p_ref the
            # flow's own least-squares fixed point is E[r/alpha], so its root lands on
            # the mean reward; log Z(q) = log E_p_ref[e^{r/alpha}] is the fixed point of
            # the *joint* solution and is out of reach during warmup (two outcomes,
            # rewards 0/1, alpha 1: flow-only optimum 0.5 versus log Z 0.6201). The
            # warmup length is therefore the fixed --subtb-flow-warmup-steps schedule.
            roots = [s["root"] for s in actor_signals if isinstance(s, dict) and s.get("root") is not None]
            rates = [s["reward_rate"] for s in actor_signals if isinstance(s, dict) and s.get("reward_rate") is not None]
            if roots and rates:
                root = sum(roots) / len(roots)
                rate = sum(rates) / len(rates)
                reference = math.log(1 - rate + rate * math.e) if 0 <= rate < 1 else float("nan")
                logger.info(
                    "SubTB warmup round %d: g(s0)=%.4f, mean reward=%.4f "
                    "(flow-only target ~ %.4f; joint target log Z(q) ~ %.4f at the pooled rate)",
                    rollout_id,
                    root,
                    rate,
                    rate,
                    reference,
                )

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        offload_train(rollout_id)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        # Never skip this in a SubTB warmup round. Under --colocate the
        # release/resume cycle hands the engines back empty weights (there is no CPU
        # weight backup), and this call is the only thing that refills them. Skipping
        # it left every warmup round after the first rolling out uniform noise: every
        # token at exactly -log(vocab_size) = -11.93, response length pinned at the
        # 8192 cap and reward 0 (job 114473), which then looked like an
        # engine-vs-trainer policy mismatch.
        if not args.critic_train_only:
            actor_model.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        # Warmup rounds cannot change the actor, so their evaluations would just
        # repeat the baseline measurement.
        if not warmup and should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args(add_custom_arguments=lambda parser: add_subtb_arguments(add_sp3o_arguments(parser)))
    validate_subtb_args(args)
    validate_sp3o_args(args)
    train(args)
