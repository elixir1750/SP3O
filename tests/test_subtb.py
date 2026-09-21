"""Mathematical and joint-gradient contracts for the NTP SubTB objective."""
from argparse import Namespace

import pytest
import torch

from slime.utils.subtb import subtb_loss, subtb_spans, validate_subtb_args, validate_subtb_sample, subtb_windows
from slime.utils.subtb import subtb_flow_warmup_active, subtb_warmup_should_stop

NUM_GPUS = 0


def test_exact_reward_tilted_tree_has_zero_loss():
    # Two first-token branches, each followed by deterministic EOS.
    ref = torch.tensor([0.3, 0.7], dtype=torch.float64)
    reward = torch.tensor([0.0, 1.0], dtype=torch.float64)
    alpha = 0.5
    z = (ref * (reward / alpha).exp()).sum()
    policy = ref * (reward / alpha).exp() / z
    for branch in range(2):
        lp = torch.stack((policy[branch].log(), policy.new_zeros(())))
        rp = torch.stack((ref[branch].log(), ref.new_zeros(())))
        flow = torch.stack((z.log(), reward[branch] / alpha))
        assert subtb_loss(lp, rp, flow, reward[branch], alpha=alpha).item() < 1e-25


def test_relative_coordinates_equal_original_flow_equation():
    lp = torch.tensor([-0.5, -1.2, -0.7], dtype=torch.float64)
    ref = torch.tensor([-0.3, -0.8, -0.9], dtype=torch.float64)
    g = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    reward = 0.7
    i, j = subtb_spans(3)
    reference_prefix = torch.cat((ref.new_zeros(1), ref.cumsum(0)))
    log_f = torch.cat((g, g.new_tensor([reward]))) + reference_prefix
    policy_prefix = torch.cat((lp.new_zeros(1), lp.cumsum(0)))
    expected = (log_f[i] + policy_prefix[j] - policy_prefix[i] - log_f[j]).square().mean()
    torch.testing.assert_close(subtb_loss(lp, ref, g, reward, sampling="legacy"), expected)


def test_separate_worker_gradients_equal_joint_gradient():
    logits = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    flow = torch.randn(5, dtype=torch.float64, requires_grad=True)
    lp = logits.log_softmax(-1)[:, 0]
    ref = torch.full((5,), -1.1, dtype=torch.float64, requires_grad=True)
    joint = subtb_loss(lp, ref, flow, 1.0)
    expected = torch.autograd.grad(joint, (logits, flow), retain_graph=True)
    actor = subtb_loss(lp, ref, flow.detach(), 1.0)
    critic = subtb_loss(lp.detach(), ref, flow, 1.0)
    actual = (torch.autograd.grad(actor, logits)[0], torch.autograd.grad(critic, flow)[0])
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
        assert torch.isfinite(a).all() and a.abs().sum() > 0
    assert ref.grad is None


def test_terminal_reward_anchors_scale_and_zero_reward_is_finite():
    lp = torch.tensor([-0.3, -0.7])
    zero = torch.zeros(2)
    assert subtb_loss(lp, lp, zero, 0).item() == 0
    assert subtb_loss(lp, lp, zero + 2, 0).item() > 0
    assert subtb_loss(lp, lp, zero, 1).item() > 0


def test_long_trajectory_span_budget_and_eos_edge():
    i, j = subtb_spans(8192, num_spans=64)
    assert len(i) == 8192 + 65
    assert ((0 <= i) & (i < j) & (j <= 8192)).all()
    assert ((i == 8191) & (j == 8192)).any()
    assert ((i == 0) & (j == 8192)).any()


def _args(**overrides):
    args = dict(loss_type="subtb_loss", subtb_alpha=1., subtb_num_spans=64,
                global_batch_size=4, rollout_batch_size=2, n_samples_per_prompt=2,
                subtb_window_size=64, subtb_num_windows=4, subtb_length_lambda=1., subtb_full_weight=.1,
                enable_weights_backuper=True, compute_advantages_and_returns=True,
                ref_update_interval=None, ref_load="fixed-reference",
                actor_num_nodes=1, critic_num_nodes=1,
                actor_num_gpus_per_node=2, critic_num_gpus_per_node=2,
                subtb_flow_inner_steps=1, subtb_flow_warmup_steps=0,
                subtb_flow_warmup_target_gap=None, subtb_flow_warmup_min_steps=2,
                num_rollout=4, use_critic=True)
    args.update(overrides)
    return Namespace(**args)


def test_valid_configuration():
    validate_subtb_args(_args())


@pytest.mark.parametrize("override", [
    dict(subtb_alpha=0), dict(subtb_alpha=float("nan")),
    dict(global_batch_size=2), dict(partial_rollout=True),
    dict(context_parallel_size=2), dict(hidden_dropout=.1),
    dict(enable_weights_backuper=False), dict(ref_update_interval=1),
    dict(num_critic_only_steps=1), dict(use_tis=True),
    dict(subtb_flow_inner_steps=0), dict(subtb_flow_warmup_steps=-1),
    dict(subtb_flow_warmup_steps=4), dict(subtb_flow_warmup_steps=2, use_critic=False),
    dict(subtb_flow_warmup_target_gap=0.0, subtb_flow_warmup_steps=2),
    dict(subtb_flow_warmup_target_gap=float("nan"), subtb_flow_warmup_steps=2),
    dict(subtb_flow_warmup_target_gap=0.05, subtb_flow_warmup_steps=0),
    dict(subtb_flow_warmup_target_gap=0.05, subtb_flow_warmup_steps=2, subtb_flow_warmup_min_steps=0),
    dict(subtb_flow_warmup_target_gap=0.05, subtb_flow_warmup_steps=2, subtb_flow_warmup_min_steps=3),
])
def test_unsupported_configs_fail_early(override):
    with pytest.raises(ValueError):
        validate_subtb_args(_args(**override))


@pytest.mark.parametrize("warmup,num_rollout,expected", [
    (0, 4, [False, False, False, False]),
    (2, 4, [True, True, False, False]),
    (4, 4, [True, True, True, True]),
])
def test_flow_warmup_rounds_are_prefix_only(warmup, num_rollout, expected):
    args = _args(subtb_flow_warmup_steps=warmup, num_rollout=num_rollout)
    assert [subtb_flow_warmup_active(args, i) for i in range(num_rollout)] == expected


def test_flow_warmup_never_applies_to_other_presets():
    for loss_type in ("policy_loss", "sft_loss", "custom_loss"):
        args = _args(loss_type=loss_type, subtb_flow_warmup_steps=2)
        assert not subtb_flow_warmup_active(args, 0)


def test_adaptive_warmup_needs_two_consecutive_converged_rounds():
    # A single lucky round must not end the warmup.
    assert not subtb_warmup_should_stop([0.04], min_steps=2, target_gap=0.05, rollout_id=0)
    assert subtb_warmup_should_stop([0.04, 0.03], min_steps=2, target_gap=0.05, rollout_id=1)
    # The second-to-last round was still far away.
    assert not subtb_warmup_should_stop([0.30, 0.03], min_steps=2, target_gap=0.05, rollout_id=1)
    # min_steps is respected even when the flow converged immediately.
    assert not subtb_warmup_should_stop([0.01, 0.01], min_steps=4, target_gap=0.05, rollout_id=1)
    assert subtb_warmup_should_stop([0.01, 0.01, 0.01, 0.01], min_steps=4, target_gap=0.05, rollout_id=3)
    # Feature disabled.
    assert not subtb_warmup_should_stop([0.0, 0.0], min_steps=2, target_gap=None, rollout_id=5)


def test_adaptive_warmup_is_opt_in_and_bounded():
    # Without a target gap the schedule stays fixed, and the fixed bound is valid.
    validate_subtb_args(_args(num_rollout=8, subtb_flow_warmup_steps=6,
                              subtb_flow_warmup_target_gap=0.05))
    validate_subtb_args(_args(num_rollout=8, subtb_flow_warmup_steps=6, subtb_flow_warmup_min_steps=6,
                             subtb_flow_warmup_target_gap=0.05))
    # The adaptive bound never exceeds the fixed warmup budget.
    validate_subtb_args(_args(subtb_flow_warmup_steps=0, subtb_flow_warmup_min_steps=2))


def _group(statuses, rewards, predictions=None):
    class FakeSample:
        def __init__(self, status, reward, prediction):
            self.status = status
            self.reward = {"score": reward, "extracted_pred": [prediction]}

        def get_reward_value(self, args):
            return self.reward["score"]

    from slime.utils.types import Sample

    if predictions is None:
        predictions = ["42"] * len(statuses)
    return [
        FakeSample(getattr(Sample.Status, s), r, p)
        for s, r, p in zip(statuses, rewards, predictions, strict=True)
    ]


def test_group_filter_drops_signal_free_groups():
    from slime.utils import subtb_filter as module

    module._state.update(budget=None, forced=0)
    subtb_group_filter = module.subtb_group_filter

    args = Namespace(reward_key="score", rollout_batch_size=6)
    # Every response hit the horizon: degenerate, 3x rollout cost, no answer.
    assert not subtb_group_filter(args, _group(["TRUNCATED"] * 4, [0.0] * 4)).keep
    assert subtb_group_filter(args, _group(["TRUNCATED"] * 4, [0.0] * 4)).reason == "all_truncated"
    # No answer could be extracted from any response: degenerate too.
    unparseable = subtb_group_filter(args, _group(["COMPLETED"] * 4, [0.0] * 4, [""] * 4))
    assert not unparseable.keep and unparseable.reason == "all_unparseable"
    # All-wrong but parseable groups are deliberately KEPT: their tilted target is
    # p_ref, so they still carry a stabilising gradient, and dropping them costs a
    # ~50% refill in every normal round.
    assert subtb_group_filter(args, _group(["COMPLETED"] * 4, [0.0] * 4)).keep
    assert subtb_group_filter(args, _group(["COMPLETED"] * 4, [1.0] * 4)).keep
    # Mixed rewards are kept, and every accepted group refunds one drop credit.
    mixed = subtb_group_filter(args, _group(["COMPLETED"] * 4, [0.0, 1.0, 0.0, 1.0]))
    assert mixed.keep and mixed.reason is None
    # Exhaust the budget on degenerate groups, then accept instead of dropping so a
    # uniformly bad region cannot starve the rollout.
    for _ in range(6):
        assert not subtb_group_filter(args, _group(["COMPLETED"] * 4, [0.0] * 4, [""] * 4)).keep
    forced = subtb_group_filter(args, _group(["COMPLETED"] * 4, [0.0] * 4, [""] * 4))
    assert forced.keep and forced.reason is None
    assert module._state["forced"] == 1
    # With the refund the filter can drop again.
    assert not subtb_group_filter(args, _group(["TRUNCATED"] * 4, [0.0] * 4)).keep
    # The zero-variance rule stays available for A/B, off by default.
    module._DROP_ZERO_STD = True
    module._state.update(budget=10, forced=0)
    assert subtb_group_filter(args, _group(["COMPLETED"] * 4, [0.0] * 4)).reason == "zero_std"
    module._DROP_ZERO_STD = False
    module._state.update(budget=None, forced=0)


@pytest.mark.parametrize("status,length,last_token,valid", [
    ("COMPLETED", 2, 99, True), ("TRUNCATED", 4, 10, True),
    ("ABORTED", 2, 10, False), ("TRUNCATED", 2, 10, False),
    ("COMPLETED", 2, 10, False), ("COMPLETED", 0, 99, False),
])
def test_terminal_convention(status, length, last_token, valid):
    sample = Namespace(status=Namespace(name=status), response_length=length, tokens=[last_token])
    if valid:
        validate_subtb_sample(sample, eos_id=99, horizon=4)
    else:
        with pytest.raises(ValueError):
            validate_subtb_sample(sample, eos_id=99, horizon=4)


def test_optimization_recovers_reference_weighted_reward_distribution():
    # Exhaustive replay of both branches; no dependency on policy sampling luck.
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    root = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    branches = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    ref = torch.tensor([.8, .2], dtype=torch.float64)
    rewards = torch.tensor([0., 1.], dtype=torch.float64)
    target = (ref.log() + rewards / .5).softmax(0)
    optimizer = torch.optim.Adam([logits, root, branches], lr=.05)
    for _ in range(300):
        optimizer.zero_grad()
        logp = logits.log_softmax(0)
        loss = sum(subtb_loss(
            torch.stack((logp[b], logp.new_zeros(()))),
            torch.stack((ref[b].log(), ref.new_zeros(()))),
            torch.cat((root, branches[b:b+1])), rewards[b], alpha=.5, num_spans=4,
        ) for b in range(2))
        loss.backward()
        optimizer.step()
    torch.testing.assert_close(logits.softmax(0), target, atol=1e-4, rtol=1e-4)
    assert loss.item() < 1e-7


@pytest.mark.parametrize("role", ["subtb_loss", "subtb_flow_loss"])
def test_megatron_loss_adapter_uses_pre_action_positions(monkeypatch, role):
    from slime.backends.megatron_utils import loss as backend

    monkeypatch.setattr(backend.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(backend.mpu, "get_tensor_model_parallel_group", lambda: None)

    def local_log_probs(logits, tokens, group, **kwargs):
        return logits.log_softmax(-1).gather(-1, tokens[:, None]), None

    monkeypatch.setattr(backend, "calculate_log_probs_and_entropy", local_log_probs)
    args = Namespace(loss_type=role, qkv_format="thd", rollout_temperature=1.,
                     subtb_alpha=1., subtb_num_spans=3, subtb_seed=7,
                     subtb_sampling="window", subtb_window_size=64, subtb_num_windows=4,
                     subtb_length_lambda=1., subtb_full_weight=.1,
                     log_probs_chunk_size=0, allgather_cp=False)
    # Two prompt tokens followed by two actions, including a terminal action.
    tokens = torch.tensor([0, 1, 2, 0])
    batch = dict(unconcat_tokens=[tokens], total_lengths=[4], response_lengths=[2],
                 values=[torch.tensor([.2, .3])], log_probs=[torch.tensor([-.5, -.8])],
                 ref_log_probs=[torch.tensor([-.7, -.6])], rewards=[1.],
                 loss_masks=[torch.ones(2)], sample_indices=[5])
    width = 1 if role == "subtb_flow_loss" else 3
    logits = torch.linspace(-.4, .9, 4 * width).reshape(1, 4, width).requires_grad_()
    actual, metrics = backend.subtb_loss_function(args, batch, logits, lambda x: x.mean())
    if role == "subtb_flow_loss":
        flow = logits[0, 1:3, 0]
        lp = batch["log_probs"][0]
    else:
        flow = batch["values"][0]
        lp = logits[0, 1:3].log_softmax(-1).gather(-1, tokens[2:, None]).squeeze(-1)
    expected = subtb_loss(lp, batch["ref_log_probs"][0], flow, 1., num_spans=3, seed=12)
    torch.testing.assert_close(actual, expected)
    assert role in metrics
    actual.backward()
    assert logits.grad[0, 1:3].abs().sum() > 0
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 3].abs().sum() == 0  # no extra terminal prediction


@pytest.mark.parametrize("length", [1, 12, 64, 65, 8192])
def test_windows_cover_terminal_and_respect_action_boundaries(length):
    windows = subtb_windows(length, seed=13)
    assert windows == subtb_windows(length, seed=13)
    assert windows[-1] == (max(0, length - 64), length)
    assert len(windows) == (1 if length <= 64 else 5)
    assert all(0 <= i < j <= length and j - i == min(64, length) for i, j in windows)


@pytest.mark.parametrize("length_lambda", [.5, 1., 1.5])
@pytest.mark.parametrize("full_weight", [0., .1, 1.])
def test_window_loss_matches_exhaustive_scalar_oracle(length_lambda, full_weight):
    lp = torch.tensor([-.2, -.8, -.4, -.7, -.9], dtype=torch.float64, requires_grad=True)
    ref = torch.tensor([-.5, -.4, -.6, -.2, -.6], dtype=torch.float64)
    g = torch.tensor([.1, .7, -.3, .2, .4], dtype=torch.float64, requires_grad=True)
    # Independent loops: keep g at internal window endpoints, reward only at T.
    states = list(g) + [g.new_tensor(1.2)]
    window_losses = []
    for start, end in subtb_windows(5, window_size=3, num_windows=4, seed=8):
        total, normalizer = 0, 0
        for i in range(start, end):
            for j in range(i + 1, end + 1):
                weight = length_lambda ** (j - i)
                total = total + weight * (states[i] + (lp[i:j] - ref[i:j]).sum() - states[j]).square()
                normalizer += weight
        window_losses.append(total / normalizer)
    full = (g[0] + (lp - ref).sum() - 1.2).square()
    expected = (1-full_weight) * torch.stack(window_losses).mean() + full_weight * full
    actual, parts = subtb_loss(lp, ref, g, 1.2, window_size=3, num_windows=4, seed=8,
                               length_lambda=length_lambda, full_weight=full_weight,
                               return_components=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(parts["full_loss"], full.detach())
    actual_grads = torch.autograd.grad(actual, (lp, g), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (lp, g))
    for a, b in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, b)


def test_terminal_window_provides_signal_at_zero_initialization():
    lp = torch.zeros(128, requires_grad=True)
    g = torch.zeros(128, requires_grad=True)
    # No random windows or full-path term: the mandatory tail still sees reward.
    loss = subtb_loss(lp, lp.detach(), g, 1., num_windows=0, full_weight=0.)
    loss.backward()
    assert loss > 0 and lp.grad[-1] != 0 and g.grad[-1] != 0
    assert g.grad[:64].abs().sum() == 0


@pytest.mark.parametrize("override", [
    dict(subtb_window_size=0), dict(subtb_num_windows=-1),
    dict(subtb_length_lambda=0), dict(subtb_length_lambda=float("nan")),
    dict(subtb_full_weight=-.1), dict(subtb_full_weight=1.1),
])
def test_invalid_window_configuration(override):
    with pytest.raises(ValueError):
        validate_subtb_args(_args(**override))


def test_zero_head_starts_learning_and_restores_checkpoint():
    from slime.backends.megatron_utils.model_provider import LinearForLastLayer

    config = Namespace(sequence_parallel=False)
    head = LinearForLastLayer(3, 1, config=config, zero_init=True)
    hidden = torch.tensor([[1., 2., 3.], [3., 1., 2.]], requires_grad=True)
    flow, _ = head(hidden)
    assert torch.count_nonzero(flow) == 0
    lp = torch.zeros(2, requires_grad=True)
    loss = subtb_loss(lp, lp.detach(), flow.flatten(), 1.)
    loss.backward()
    assert head.weight.grad.abs().sum() > 0
    assert head.bias.grad.abs().sum() > 0
    assert hidden.grad.abs().sum() == 0  # first step only: W is initially zero
    assert lp.grad.abs().sum() > 0
    optimizer = torch.optim.SGD(head.parameters(), lr=.01)
    optimizer.step()
    optimizer.zero_grad()
    hidden.grad.zero_()
    flow, _ = head(hidden)
    subtb_loss(lp.detach(), lp.detach(), flow.flatten(), 1.).backward()
    assert hidden.grad.abs().sum() > 0

    restored = LinearForLastLayer(3, 1, config=config, zero_init=True)
    restored.load_state_dict(head.state_dict())
    torch.testing.assert_close(restored(hidden)[0], head(hidden)[0])
    assert torch.count_nonzero(restored.weight) > 0


def test_default_ppo_head_remains_random_and_zero_reward_has_no_subtb_gradient():
    from slime.backends.megatron_utils.model_provider import LinearForLastLayer

    config = Namespace(sequence_parallel=False)
    assert LinearForLastLayer(8, 1, config=config).weight.count_nonzero() > 0
    head = LinearForLastLayer(3, 1, config=config, zero_init=True)
    lp = torch.tensor([-.2, -.5], requires_grad=True)
    flow, _ = head(torch.ones(2, 3))
    loss = subtb_loss(lp, lp.detach(), flow.flatten(), 0.)
    loss.backward()
    assert loss.item() == 0
    assert lp.grad.abs().sum() == 0
    assert head.weight.grad.abs().sum() == 0
