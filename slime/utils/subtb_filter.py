"""Group filter for NTP SubTB training: drop degenerate groups only.

Wired through slime's dynamic-sampling hook
(``--dynamic-sampling-filter-path slime.utils.subtb_filter.subtb_group_filter``).
Dropped groups are refilled from the data source, so the trained batch stays
``rollout_batch_size x n_samples_per_prompt``; each drop is recorded as
``rollout/dynamic_filter/drop_<reason>``.

"Degenerate" means the policy produced no usable answer for any of the samples:

* ``all_truncated`` - every response hit the response-action horizon.
* ``all_unparseable`` - every response's answer extraction is empty, i.e. the
  reward function could not even read an answer.

Measured on the dumped pilot rounds: normal rounds contain 0% all-truncated and
0% all-unparseable groups, while a degenerate round (mean response 8002 of 8192)
had 66-69% all-truncated and 100% all-unparseable groups. The rule therefore
costs nothing in normal rounds and removes exactly the rows that burn the most
rollout compute without carrying an answer.

An earlier revision also dropped groups whose rewards were all identical. The
dumps show that fires on 41-59% of the groups in *every* normal round (mixed
rewards are only ~50%), which doubles rollout cost per round for roughly the same
signal per GPU-hour, and it removes both "all wrong but parseable" groups (whose
tilted target is legitimately p_ref, so they still carry a stabilising gradient)
and "all correct" groups (a real tilt signal). It is kept behind
``SUBTB_FILTER_ZERO_STD=1`` for A/B use only.

The filter is *bounded*: a drop budget stops it from starving a rollout when the
policy is uniformly bad on a region of the data (measured on job 114344, where a
degenerate region produced extraction-empty, all-zero groups indefinitely and the
rollout generated thousands of responses without filling its batch). The budget
starts at one rollout batch worth of drops and gains one credit per accepted
group, so the sustained drop ratio is capped at ~1/2 and every rollout still
fills. Groups accepted because the budget ran out are logged as "group filter
saturated"; keeping them is not harmful for NTP SubTB, they simply contribute
value calibration without a reward tilt.
"""
import logging
import os

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_DROP_CREDIT_PER_ACCEPT = 1
_DROP_ZERO_STD = os.environ.get("SUBTB_FILTER_ZERO_STD", "0") == "1"
_state = {"budget": None, "forced": 0}


def _unparseable(sample) -> bool:
    """True when the reward function could not extract any answer."""
    reward = getattr(sample, "reward", None)
    if not isinstance(reward, dict):
        return False
    prediction = reward.get("extracted_pred")
    if prediction is None:
        return False
    if isinstance(prediction, (list, tuple)):
        return not prediction or all(not str(item).strip() for item in prediction)
    return not str(prediction).strip()


def subtb_group_filter(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    if _state["budget"] is None:
        _state["budget"] = int(getattr(args, "rollout_batch_size", 64) or 64)

    reason = None
    if all(sample.status == Sample.Status.TRUNCATED for sample in samples):
        reason = "all_truncated"
    elif all(_unparseable(sample) for sample in samples):
        reason = "all_unparseable"
    elif _DROP_ZERO_STD:
        rewards = {round(sample.get_reward_value(args), 6) for sample in samples}
        if len(rewards) == 1:
            reason = "zero_std"

    if reason is not None and _state["budget"] > 0:
        _state["budget"] -= 1
        return DynamicFilterOutput(keep=False, reason=reason)

    _state["budget"] += _DROP_CREDIT_PER_ACCEPT
    if reason is not None:
        _state["forced"] += 1
        if _state["forced"] == 1 or _state["forced"] % 32 == 0:
            logger.warning(
                "SubTB group filter saturated: accepting a %s group after %d forced accepts "
                "(drop budget exhausted)", reason, _state["forced"])
    return DynamicFilterOutput(keep=True)
