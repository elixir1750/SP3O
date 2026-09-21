# NTP Subtrajectory Balance

This experimental preset reuses the rollout engine and separate actor/critic
training infrastructure, but replaces PPO with SubTB. There is no GAE, PPO
ratio clipping, clipped value regression, or critic-only warmup. The model
stored under `critic/` is a flow estimator, not an expected-return critic.

## Target and state space

For each prompt q, generate tokens until EOS or exactly H response actions,
where H is `MAX_RESPONSE_LEN`. H is an explicit terminal horizon of this task,
not an unobserved continuation whose return is silently bootstrapped. An EOS
action is included in the probability; no EOS is invented at the length limit.
The reference is the fixed `MEGATRON_CHECKPOINT` supplied as `--ref-load`.

The desired distribution over these bounded terminal sequences is

    p*(x | q) = p_ref(x | q) exp(r(q,x) / alpha) / Z(q).

At the horizon, p_ref is the probability of that prefix, with no extra stop
probability. Every nonterminal state is a full token prefix, so there is one
parent and the backward transition probability is 1. Raw scalar reward r is
not centered or group-normalized. Zero rewards are valid because the objective
uses log R = log p_ref + r/alpha, without computing log r or exp(r/alpha).

## Parameterization and objective

New SubTB runs zero-initialize the scalar head weights and bias, keeping the
pretrained backbone. Actor and flow train jointly from the first batch; no
warmup is used. Set `SUBTB_FLOW_INIT=random` to reproduce older runs. Loading
a trained checkpoint restores its learned head after construction, rather than
resetting it. PPO/SP3O scalar-head initialization is unaffected.

The scalar head predicts

    g(s) = log F(s) - log p_ref(prefix(s) | q).

This is a relative log-flow; the root has g(s0) = log Z(q). On a terminal
state, substitute g(x) = r(q,x)/alpha instead of a network prediction.
For each selected subtrajectory [i,j], minimize the squared residual

    g(si) + sum_{t=i}^{j-1} [log pi(at|st) - log p_ref(at|st)] - g(sj).

This is exactly the original SubTB residual after the change of coordinates.
Scalar outputs at token t see the prefix BEFORE action t. A T-token response
requires T network outputs, with its T+1st state provided by the terminal
boundary. The flow head is unconstrained; it predicts a logarithm.

## Window sampling (default)

For T > 64, sample four length-64 action windows uniformly by start position,
with replacement, then append the mandatory terminal window [T-64,T].
Overlapping/duplicate windows retain their sampling multiplicity. For T <= 64,
use [0,T] once. Every window has 65 boundary states when its action length is 64.
Only the true trajectory terminal is replaced by r/alpha; internal window
endpoints use the network output, never the terminal reward.

Within each window [b,e], enumerate every b <= i < j <= e and compute

    L_window = sum(lambda**(j-i) * residual(i,j)**2) / sum(lambda**(j-i)).

The normalized weights are computed with a softmax over length*log(lambda),
so extreme positive lambda values do not overflow. There are 2080 intervals
in a 64-action window. Default lambda=1 gives equal weight per interval, not
per length. Average window losses equally, then combine with the complete path:

    L_trajectory = (1-eta) * mean(L_window) + eta * residual(0,T)**2.

Default eta=0.1 is an initial experimental choice, not a tuned or paper-reported
optimum. Thus 90% of the coefficient goes to windows, 10% to the complete path;
that is a loss coefficient and not a gradient share - the gradient split follows
the residual magnitudes, and a single full-path term carries a whole trajectory's
error while a window term carries 64 actions, so 0.1 does not mean "the full path
contributes 10% of the gradient". The full path may also appear inside a window
for short responses, by design. A 64-action window also cannot constrain
mid-range sub-trajectories directly; those are linked through the full-path term
and through overlapping windows.

Window selection uses a deterministic seed plus sample index shared by actor
and flow workers. Extra loss storage is O(T + K*W**2); model forward passes
still require the full causal prefix. This is a window-weighted objective,
not an unbiased estimator of all full-trajectory subintervals. Sequence losses
are averaged equally in each global training batch.

For reproducing the original smoke (113450), use `SUBTB_SAMPLING=legacy`:
all T single edges, the full path, and `SUBTB_NUM_SPANS=64` random intervals,
with a flat average over occurrences. Window sampling replaces that default.

W&B records the combined loss and its unweighted components separately:
`train/subtb_loss`, `train/subtb_window_loss`, `train/subtb_full_loss`, and the
corresponding `train/critic-...` metrics. Optimizer gradients, rewards, response
lengths and mean relative flow are also recorded by the existing logging path.

## Joint update

Before each update, workers cache and exchange current policy log probabilities,
reference log probabilities and current flow predictions. The actor computes
its gradient using detached flow predictions; the flow model computes its
gradient using detached policy probabilities. The actor takes exactly one
optimizer step per rollout.

Ordering, stated precisely: the two models are separate processes (actor on its
own GPUs, flow on its own) that exchange data once per rollout, so within a
rollout the update is simultaneous rather than sequential —
`theta_{k+1} = theta_k - eta * grad L(theta_k, phi_k)` while the flow takes its
`K` steps on the same cached `theta_k` snapshot to reach `phi_{k+1}`. The actor's
gradient therefore uses the flow *before* this rollout's flow steps; the flow
leads only from the next rollout on. Making the actor consume `phi_{k+1}` in the
same rollout needs a flow-first ordering with an extra sync (the flow would have
to re-evaluate `g` after its steps and publish it, and the actor would then have
to wait for that), which serialises the two blocks instead of overlapping them;
worth doing deliberately, not as a side effect.

Fit the flow before letting the actor move. With a zero-initialised flow and a
single flow step per rollout, the actor moves the per-trajectory log-ratio
`sum_t log pi/p_ref` by roughly one nat in one step, while the flow's scalar
head moves by order 1e-3 nat per step: the baseline lags by two orders of
magnitude and the residual is dominated by the un-baselined log-ratio instead of
the reward tilt. `--subtb-flow-warmup-steps W` therefore keeps the actor frozen
for the first `W` rollouts (its reference/actor forwards still run and still
publish log-probs, and it still re-synchronises its unchanged weights to the
rollout engines; only its optimizer step is skipped), and
`--subtb-flow-inner-steps K` lets the flow take `K`
optimizer steps per rollout on that rollout's cached actor snapshot. Because the
flow does not influence sampling, those extra steps need no extra rollouts and
no actor-side replay. This is a two-timescale scheme, not joint SGD: the loss is
the same, gradient clipping and the two optimizers still act independently, and
the actor remains strictly on-policy with one step per rollout.

During warmup the actor is exactly the reference, so the flow loss degenerates
to a consistency regression whose fixed point is `g(s) -> E[r/alpha | s]`: the
root lands on the observed mean reward, not on
`log Z(q) = log E_p_ref[exp(r/alpha)]`, which is the fixed point of the joint
solution and is only reached once the actor starts moving. `train/subtb_root_value`
reports `g(s0)` and the warmup rounds log both scales side by side; see
"What the warmup can and cannot converge to" below.

### Warmup still has to refill the rollout engines

Under `--colocate` the engines share the training GPUs, so every round releases
their memory (`release_memory_occupation`) for the trainer and resumes it
afterwards. With `--sglang-enable-weights-cpu-backup=False` the released weights
are *not* restored by `resume_memory_occupation`; the contract in
`slime/ray/rollout.py` is that `update_weights` re-pushes them before the next
rollout. Skipping that sync during warmup therefore leaves the engines holding
empty weights: the forward pass returns constant logits, the sampler draws from
a uniform distribution over the vocabulary, and the round collapses into
multi-language token soup that runs to the 8192-token cap with reward 0.

That failure is unambiguous in the metrics rather than subtle: every token's
engine log-prob is exactly `-log(vocab_size)` (= `-11.93` for Qwen3's 151,936
tokens, a float32 constant to 9 digits), so
`rollout/rollout_log_probs` is pinned to that value while the actor's
recomputation of the same tokens reports about `-13.2`. It is *not* an
engine/trainer numerics gap, so do not "fix" it by shrinking batches, response
limits or tolerating a mismatch: check the weight sync first. Job 114473 spent
45 minutes per round generating that soup, and the earlier "degenerate rounds"
in 113925 were the same bug.

`rollout/engine_logprob_gap` (engine minus actor mean log-prob, token-weighted)
is logged every round: healthy rounds sit at ~1e-3 nats, an empty engine lands
near 1 nat. `--subtb-engine-gap-abort 0.1` turns it into a hard stop in the
driver, so a run aborts after one bad round instead of training on noise for
hours.

`train_actor` must end with
`weights_backuper.backup("actor")`. The backup is what the next round restores
when it switches back from "ref" to "actor" (`_switch_model` -> `restore`) and
what `update_weights` reads to fill the engines. An early `return` in that
function (one revision had it, which also left a `rollout_id` reference in the
helper that swallowed the tail) makes every joint optimizer step evaporate and
keeps serving the initial policy. Note that the engine/actor gap does *not* see
this one: engines and actor are both reverted to the same stale copy, so they
agree with each other while the policy stands still. The signature is
`rollout/actor_ref_logprob_drift`, the mean `log_probs - ref_log_probs`, which
stays at exactly 0 in every joint round instead of growing. `tests/test_subtb.py`
asserts the code shape statically (tail reachable + backup present + no method
reading a name that only exists elsewhere).

Degenerate groups are dropped before they reach the objective
(`--dynamic-sampling-filter-path slime.utils.subtb_filter.subtb_group_filter`): a
group is discarded when every response is TRUNCATED or when no response yields a
parsable answer. Measured over the dumped rounds, normal rounds contain 0%
all-truncated and 0% all-unparseable groups, while a degenerate round (mean
response 8002 of 8192) had 66-69% all-truncated and 100% all-unparseable, so this
rule costs nothing in normal rounds and removes exactly the rows that burn the
most rollout compute without carrying an answer. The data source refills the
batch, and a bounded drop budget (one batch worth of drops, one credit refunded
per accepted group, forced accepts logged as "group filter saturated") keeps a
uniformly bad region from starving the rollout.

An earlier revision also dropped groups whose rewards were all identical. The
dumps show that rule fires on 41-59% of the groups in *every* normal round, which
doubles rollout cost per round for roughly the same signal per GPU-hour, and it
removes both all-wrong-but-parseable groups (whose tilted target is legitimately
p_ref, so they still carry a stabilising gradient) and all-correct groups (a real
tilt signal). It is therefore off by default and reachable only through
`SUBTB_FILTER_ZERO_STD=1` for A/B work.

### What the warmup can and cannot converge to

The warmup length is the fixed `--subtb-flow-warmup-steps` schedule. An earlier
revision tried to end it early by demanding `|mean(g(s0)) - log Z(q)| <= gap` for
two consecutive rounds, and that criterion is wrong: with the actor frozen at
`p_ref` every log-ratio term vanishes, so the flow's own least-squares fixed point
is `g(s) -> E[r/alpha | s]` and the root lands on the **mean reward**. `log Z(q)`
is the fixed point of the *joint* solution, which a frozen actor cannot reach. A
two-outcome example (rewards 0 and 1, `alpha = 1`, uniform `p_ref`) makes the gap
irreducible: minimising `0.5 g^2 + 0.5 (g-1)^2` gives `g* = 0.5`, while
`log Z = log((1 + e)/2) = 0.6201`, so `|g* - log Z| = 0.12 > 0.05` at the exact
optimum. Pooling the reward rate before taking the log
(`log E_q[Z(q)] != E_q[log Z(q)]`) adds a second, smaller error.

Warmup is therefore a scale initialisation: it lifts the root from 0 to the
order of the mean reward before the actor starts moving, and the joint phase then
pulls `g` up to `log Z(q)` as the actor's log-ratios enter the residual. Both
scales are logged every warmup round (the actor reports `root`, `reward_rate`,
`mean_reward_target` and `log_z_joint_target`) and neither is used as a gate.
If the warmup is ever made adaptive, the quantity to watch is the flow-only loss
on a fixed validation sample (plateau), not the distance to the joint target.

The first implementation enforces one global batch per rollout, matching
actor/flow topology, Megatron, CP=PP=1, temperature 1, zero dropout, complete
unmasked responses, and a fixed reference. Partial rollout, PPO TIS and moving
reference updates are disabled. `num_critic_only_steps` must stay zero: that
knob also removes the actor forwards and the actor/flow sync the flow objective
depends on. The preset does not add actor-side replay or MTP actions, and claims
no convergence. Other PPO/GRPO/SP3O presets retain their existing behavior.

## Launch

Use the existing environment/path configuration in this directory. From a
Slurm allocation (never directly on a LUMIA login/storage host):

```bash
export NUM_STEPS_PER_ROLLOUT=1
export SUBTB_ALPHA=1.0 SUBTB_SAMPLING=window SUBTB_FLOW_INIT=zero
export SUBTB_WINDOW_SIZE=64 SUBTB_NUM_WINDOWS=4
export SUBTB_LENGTH_LAMBDA=1.0 SUBTB_FULL_WEIGHT=0.1
# Two-timescale schedule: fit the flow first, then keep it ahead of the actor.
export SUBTB_FLOW_WARMUP_STEPS=4 SUBTB_FLOW_INNER_STEPS=4
export WANDB_MODE=online WANDB_PROJECT=SP3O-SubTB
# Optional: export WANDB_ENTITY=your-team
bash examples/sp3o/train_subtb.sh
```

`NUM_ROLLOUT` must exceed `SUBTB_FLOW_WARMUP_STEPS`; the leading warmup rounds
take `K` flow steps each and the remaining rounds take one actor step plus `K`
flow steps. The preset forces zero critic-only warmup. Default batch: 64
prompts, 8 responses each, a single 512-response optimizer step for the actor
per joint rollout. `--disable-weights-backuper` is incompatible: fixed reference
weights must be preserved. Account for the extra host memory of
actor/reference backups.

Leave the rollout engine at its defaults. The upstream Qwen3-4B example passes
neither `--sglang-max-running-requests` nor `--sglang-server-concurrency`: SGLang
sizes the running-request cap from its own KV pool (here `min(pool_tokens/2,
4096) = 4096`, with the pool being the real limit) and slime keeps its default
client concurrency. Pinning the engine cap far below the offered concurrency
(for example 16 running requests per engine while the client may hold hundreds
in flight) starves throughput and, if the client concurrency is left at its
default, backs requests up in the router until it answers HTTP 503
`no_available_workers`. Cap a setting only with its partner: engine capacity and
client concurrency have to move together, and any cap should be chosen from a
measured sweep rather than from a worst-case token budget.

Unit tests in `tests/test_subtb.py` check an analytically solved tree,
coordinate equivalence, split versus joint gradients, terminal anchoring,
window boundary coverage, exhaustive scalar loss/gradient equivalence,
terminal-window signal at zero predictions, and unsupported-config rejection.

On the prepared LUMIA environment, the local smoke workflow accepts:

```bash
sbatch --job-name=subtb-4B-smoke --mem=384G local/lumia/scripts/smoke.sbatch 4B subtb
```

It uses four Ada GPUs, three batches of four short responses, fixed seed 1234,
real math reward, held-out smoke evaluation, and both model checkpoints.
Inspect the job status and `smoke-evidence.json`; submission is not success.

SubTB enables W&B by default, using existing login credentials; it does not
create or modify credentials. Online runs require W&B authentication visible to
the Slurm process. To explicitly record offline, set WANDB_MODE=offline in a
custom launch. Do not put API keys in tracked files or command-line arguments.

## Evaluating learning (distinct from smoke validation)

Primary metric: mean correctness on a fixed held-out question set, comparing
initial Qwen3-4B-Base and checkpoints at identical response budgets, temperatures,
verifiers and numbers of samples. An initial pilot can use 256 held-out DAPO
questions and 4 generations per question, with no prompt overlap with training.
Keep a separate final benchmark (e.g. MATH500); do not tune on the final test set.
The LUMIA pilot split and three-round TP4 capacity run have been validated;
the 100-round efficacy job was submitted. This is not evidence of convergence.
See [SUBTB_PILOT.md](SUBTB_PILOT.md) to reproduce the configuration elsewhere.

Report average correctness (not pass@4), paired per-question improvements and
uncertainty. Multiple generations of the same question are correlated; estimate
uncertainty over questions. Confirm promising changes across training seeds.
Track truncation, response lengths, repetition, training reward, and sampling
cost alongside accuracy. Compare zero and random head initialization at matched
training/rollout budgets. SubTB loss alone does not establish better reasoning.

For optional flow diagnostics, hold fixed a small panel of reference-generated
prefixes and estimate log(mean(exp(reward/alpha))) from independent reference
continuations. Track held-out flow error against that noisy MC estimate; this
is diagnostic only, not a training target in the no-warmup baseline.

With zero initialization, actor=reference and all-zero rewards, the SubTB
objective and its gradients can be exactly zero. The execution smoke allows
this while testing positive-reward gradient activation separately in unit
tests. Optimizer weight decay and numerical differences can still cause drift;
nonzero gradients alone are not evidence of reward-driven improvement.
