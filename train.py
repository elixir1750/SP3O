import logging

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action
from slime.utils.subtb import (
    add_subtb_arguments,
    subtb_flow_warmup_active,
    subtb_warmup_should_stop,
    validate_subtb_args,
)
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
    warmup_gap_history: list[float] = []
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

        # Adaptive SubTB warmup: once the flow's root prediction sits within the
        # requested gap of log Z(q) for consecutive rounds, start the joint phase
        # early. --subtb-flow-warmup-steps stays the upper bound.
        if warmup and actor_signals and args.subtb_flow_warmup_target_gap is not None:
            gaps = [s["gap"] for s in actor_signals if isinstance(s, dict) and s.get("gap") is not None]
            if gaps:
                mean_gap = sum(gaps) / len(gaps)
                warmup_gap_history.append(mean_gap)
                if subtb_warmup_should_stop(
                    warmup_gap_history,
                    min_steps=args.subtb_flow_warmup_min_steps,
                    target_gap=args.subtb_flow_warmup_target_gap,
                    rollout_id=rollout_id,
                ):
                    # The run is sized as (warmup bound + planned joint rounds), so
                    # ending the warmup early already turns the unused warmup rounds
                    # into extra joint rounds: the planned joint budget is a floor.
                    saved = max(0, args.subtb_flow_warmup_steps - (rollout_id + 1))
                    args.subtb_flow_warmup_steps = rollout_id + 1
                    actor_model.set_subtb_warmup_steps(rollout_id + 1)
                    logger.info(
                        "SubTB warmup converged (mean root gap %.4f): joint phase starts at rollout %d "
                        "(%d warmup rounds saved become joint rounds)",
                        mean_gap,
                        rollout_id + 1,
                        saved,
                    )

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        offload_train(rollout_id)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        if not args.critic_train_only and not warmup:
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
