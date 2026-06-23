# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from ..protocol import DataProto


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    result = {}
    for key, value in metrics.items():
        try:
            # Convert to numpy array and check if scalar numeric
            arr = np.asarray(value)
            if arr.size == 0:
                continue
            if not np.issubdtype(arr.dtype, np.number):
                continue
            result[key] = float(np.mean(arr))
        except (TypeError, ValueError):
            # Skip values that cannot be converted to numeric arrays
            continue
    return result


def compute_length_metrics(batch: DataProto) -> dict[str, Any]:
    max_response_length = batch.batch["responses"].size(-1)
    max_prompt_length = batch.batch["attention_mask"].size(-1) - max_response_length

    prompt_length = batch.batch["attention_mask"][:, :-max_response_length].sum(-1).float()
    response_length = batch.batch["attention_mask"][:, -max_response_length:].sum(-1).float()

    return {
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.eq(response_length, max_response_length).float().mean().detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.eq(prompt_length, max_prompt_length).float().mean().detach().item(),
    }


def _compute_entropy_metrics(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    """Compute entropy-related metrics from log probabilities.

    Uses ``-log_prob(sampled_token)`` as an unbiased estimator of the token-level
    Shannon entropy ``H(p) = -∑ p(v) log p(v)`` (valid when the sampled token is
    drawn from the policy distribution itself).

    Args:
        log_probs:     shape (batch, response_length). Token log-probs under the
                       policy being measured (e.g. old_log_probs or ref_log_probs).
        response_mask: Boolean/float mask of the same shape – True for response tokens.
        prefix:        Metric name prefix, e.g. ``"policy"`` or ``"ref"``.

    Returns:
        Dictionary with the following keys (all floats):

        ``{prefix}/entropy/mean``
            Mean per-token entropy estimate over all valid (non-masked) tokens.
            Primary indicator of **entropy collapse**: monotonically decreasing →
            the model is becoming over-confident and losing diversity.

        ``{prefix}/entropy/std``
            Std of per-token entropy values across valid tokens.
            Low std = uniform confidence; high std = mixed confident/uncertain.

        ``{prefix}/entropy/min``
            Minimum per-token entropy (most confident token position across batch).

        ``{prefix}/entropy/max``
            Maximum per-token entropy (least confident token position across batch).

        ``{prefix}/entropy/per_seq_mean``
            Average of each sequence's mean token entropy (mean-of-means).
            Normalises out sequence-length effects; useful to compare across
            batches with different length distributions.

        ``{prefix}/entropy/per_seq_std``
            Std across sequences of their per-sequence entropy.
            Low → model has similar confidence on all prompts;
            High → some prompts still uncertain while others are already collapsed.
    """
    # neg_log_probs[i,t] ≈ H(p_{i,t})  (token-level entropy estimator)
    neg_lp = -log_probs  # (batch, response_length)
    mask = response_mask.bool()

    valid_neg_lp = neg_lp[mask]  # 1-D tensor of valid token entropies

    # per-sequence mean entropy
    seq_lengths = mask.sum(dim=-1).float().clamp(min=1)           # (batch,)
    seq_entropy = (neg_lp * mask.float()).sum(dim=-1) / seq_lengths  # (batch,)

    return {
        f"{prefix}/entropy/mean":         valid_neg_lp.mean().item(),
        f"{prefix}/entropy/std":          valid_neg_lp.std().item(),
        f"{prefix}/entropy/min":          valid_neg_lp.min().item(),
        f"{prefix}/entropy/max":          valid_neg_lp.max().item(),
        f"{prefix}/entropy/per_seq_mean": seq_entropy.mean().item(),
        f"{prefix}/entropy/per_seq_std":  seq_entropy.std().item(),
    }


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].size(-1)
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # ── 熵指标 ──────────────────────────────────────────────────────────────
    # old_log_probs: 本步 rollout 时当前策略的 token log-prob（更新前），
    #   监控这个可以最早发现 entropy collapse。
    # ref_log_probs: 参考策略（SFT 基线）的 token log-prob（可选）。
    # 两者的熵差 = policy entropy − ref entropy，正值表示策略比基线更分散，
    #   负值表示策略比基线更收敛（entropy collapse 的早期信号）。
    entropy_metrics: dict[str, float] = {}
    if "old_log_probs" in batch.batch:
        # response_mask shape 与 old_log_probs shape 需要对齐
        old_lp = batch.batch["old_log_probs"]   # (batch, response_length)
        resp_mask = batch.batch.get("response_mask", response_mask)
        # 确保 mask 形状与 log_probs 一致
        if resp_mask.shape != old_lp.shape:
            resp_mask = response_mask
        entropy_metrics.update(_compute_entropy_metrics(old_lp, resp_mask, "policy"))

        if "ref_log_probs" in batch.batch:
            ref_lp = batch.batch["ref_log_probs"]  # (batch, response_length)
            entropy_metrics.update(_compute_entropy_metrics(ref_lp, resp_mask, "ref_policy"))

            # entropy gap: policy − ref  (负值 → policy 比 ref 更 collapsed)
            neg_old_lp = -old_lp
            neg_ref_lp = -ref_lp
            mask_f = resp_mask.bool()
            valid_gap = (neg_old_lp - neg_ref_lp)[mask_f]
            entropy_metrics["policy/entropy/gap_vs_ref_mean"] = valid_gap.mean().item()
            entropy_metrics["policy/entropy/gap_vs_ref_std"]  = valid_gap.std().item()

    return {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        **entropy_metrics,
        **compute_length_metrics(batch),
    }


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
    num_overall_tokens = sum(batch.meta_info["global_token_num"])
    num_tokens_of_section = {
        **dict.fromkeys(["gen", "reward"], num_response_tokens),
        **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
    }
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], num_gpus: int) -> dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * num_gpus),
    }


def _compute_group_stats(
    uid2scores: dict[str, list[float]],
    prefix: str,
    success_threshold: float = 0.9,
    zero_var_eps: float = 1e-4,
    action_type_weight: float = 0.5,
) -> dict[str, Any]:
    """Core computation of rollout-group statistics. Single source of truth.

    Given a mapping from prompt-uid to the list of rollout scores in that group,
    computes a comprehensive set of metrics covering:

    ① Backward-compatible basics (all_fail / all_success / avg_success_rate …)
    ② Zero-variance detection — the *true* GRPO dead zone (advantage = 0)
    ③ Fine-grained all-above / all-below threshold breakdowns
    ④ Group distribution stats (min / std per group)
    ⑤ Rollout-level effective-training-signal ratio
    ⑥ Individual score distribution (semantically binned by action_type_weight)

    Args:
        uid2scores: Dict mapping uid → list of per-rollout scores in that group.
        prefix:     WandB metric key prefix (e.g. ``"rollout"`` or ``"val"``).
        success_threshold: Score strictly above this is considered "successful".
            Default 0.9. With default weights 0.5/0.5, the only score above 0.9
            is 1.0 (full credit), so this effectively means "perfect".
        zero_var_eps: Groups whose score std is below this are treated as
            zero-variance (no GRPO gradient). Default 1e-4.
        action_type_weight: Weight assigned to the action-type sub-score (e.g.
            0.5 or 0.2). Used to define **semantically correct** score-dist bins
            that remain meaningful regardless of the weight setting.
            When ``action_type = 1, action_params = 0`` the overall score equals
            exactly ``action_type_weight``, which is the bin boundary between
            "type only" and "partial params".

    Returns:
        Dictionary of all group metrics, described below.

        ── Basics (backward-compatible) ──────────────────────────────────────
        ``all_fail_count/ratio``    No rollout in this group scored > threshold.
                                    Note: includes mixed {0,w_t} groups which DO
                                    have gradient – use zero_variance for dead zones.
        ``all_success_count/ratio`` All rollouts scored > threshold (= all perfect).
        ``any_success_ratio``       = 1 − all_fail_ratio.
        ``avg_success_rate``        Mean of (# successes / n) across groups.
        ``group_max_score/mean``    Mean of each group's max score.
        ``group_mean_score/mean``   Mean of each group's mean score.

        ── Zero variance (true GRPO dead zones) ──────────────────────────────
        ``zero_variance_count/ratio``  Groups with std < zero_var_eps → advantage≈0.
        ``has_gradient_ratio``         = 1 − zero_variance_ratio.

        ── Zero-variance 内部 4 路分解 (over total_prompts) ─────────────────
        Zero-variance groups are further classified by their uniform score level.
        The four classes are **mutually exclusive and exhaustive**:
          zv_wrong + zv_type_only + zv_partial + zv_perfect = zero_variance_count

        ``zv_wrong_count/ratio``      Uniform score <  lo (= w_t×0.5)
                                       → action type totally wrong; model can't
                                         produce correct action type at all.
        ``zv_type_only_count/ratio``  Uniform score ∈ [lo, hi] (= w_t×1.1)
                                       → action type right but params always 0;
                                         stuck at w_t, no params learning.
        ``zv_partial_count/ratio``    Uniform score ∈ (hi, 0.99)
                                       → type right + partial params, but locked
                                         at the same intermediate score every time.
        ``zv_perfect_count/ratio``    Uniform score ≥ 0.99
                                       → model fully saturated on this prompt.

        ── Zero-variance 内部相对比例 (within zero_variance) ──────────────────
        Normalized by zero_variance_count (only present when zero_var_count > 0).
        ``zv_composition/all_wrong``     = zv_wrong_count    / zero_variance_count
        ``zv_composition/type_only``     = zv_type_only_count/ zero_variance_count
        ``zv_composition/partial_stuck`` = zv_partial_count  / zero_variance_count
        ``zv_composition/all_perfect``   = zv_perfect_count  / zero_variance_count
        Sum = 1.0 (within dead zones, how much is wrong / stuck-type / stuck-partial / saturated).

        ── Fine-grained dead-zone cross-check (legacy keys) ─────────────────
        ``all_zero_count/ratio``          All scores < 0.10  (≡ zv_wrong).
        ``all_perfect_count/ratio``       All scores ≥ 0.99  (≡ zv_perfect).
        ``all_type_only_max_count/ratio`` max_score ≤ action_type_weight*1.1
                                          → no rollout scored any params credit;
                                          = zv_wrong + zv_type_only.

        ── All-above threshold breakdowns (min_score > X) ────────────────────
        ``all_above_70_ratio``  All rollouts scored > 0.70
        ``all_above_80_ratio``  All rollouts scored > 0.80
        ``all_above_90_ratio``  All rollouts scored > 0.90  (≡ all_success_ratio
                                when success_threshold=0.9)

        ── All-below threshold breakdowns (max_score < X) ────────────────────
        ``all_below_10_ratio``  All rollouts scored < 0.10
        ``all_below_20_ratio``  All rollouts scored < 0.20
        ``all_below_30_ratio``  All rollouts scored < 0.30

        ── Group distribution stats ──────────────────────────────────────────
        ``group_min_score/mean``  Mean of each group's minimum score.
        ``group_std/mean``        Mean of each group's score std.

        ── Rollout-level effective training signal ────────────────────────────
        ``effective_rollout_ratio``  Fraction of individual rollouts that belong
                                     to a group with non-zero variance (i.e., the
                                     rollout may receive a non-zero GRPO advantage).

        ── Individual rollout score distribution (weight-aware 4 bins) ────────
        Bins are derived from ``action_type_weight`` (w_t) so they map directly
        to the semantic meaning of each score level regardless of weight setting.

        Let  lo = w_t * 0.5,   hi = w_t * 1.1 :

        ``score_dist/type_wrong``   score < lo
                                    → overall ≈ 0.0: action type mismatched
        ``score_dist/type_only``    lo ≤ score ≤ hi
                                    → overall ≈ w_t: type right, params = 0
        ``score_dist/partial``      hi < score < 0.99
                                    → type right, params partially correct
        ``score_dist/perfect``      score ≥ 0.99
                                    → overall ≈ 1.0: fully correct

        Examples
        ─────────────────────────────────────────────────────────────────────
        Weights 0.5/0.5 → score space {0.0, 0.5, 0.75, 0.9, 1.0}
          lo=0.25  hi=0.55
          0.0  → type_wrong ✓   0.5  → type_only ✓
          0.75 → partial ✓      0.9  → partial ✓    1.0 → perfect ✓

        Weights 0.2/0.8 → score space {0.0, 0.2, 0.6, 0.84, 1.0}
          lo=0.10  hi=0.22
          0.0  → type_wrong ✓   0.2  → type_only ✓
          0.6  → partial ✓      0.84 → partial ✓    1.0 → perfect ✓
    """
    total_prompts = len(uid2scores)
    if total_prompts == 0:
        return {}

    # ── score_dist bin thresholds (weight-aware) ──────────────────────────────
    # lo: upper boundary of "type wrong" region  (score = 0.0 if type mismatched)
    # hi: upper boundary of "type only"  region  (score = w_t if type right, params=0)
    _dist_lo = action_type_weight * 0.5   # 0.25 for 0.5/0.5 | 0.10 for 0.2/0.8
    _dist_hi = action_type_weight * 1.1   # 0.55 for 0.5/0.5 | 0.22 for 0.2/0.8

    # ── group-level counters ──────────────────────────────────────────────────
    all_fail_count = 0
    all_success_count = 0
    zero_var_count = 0
    all_zero_count = 0
    all_perfect_count = 0
    all_type_only_max_count = 0   # max_score ≤ _dist_hi → no params credit at all
    # min_score > threshold → all_above;  max_score < threshold → all_below
    _above = {70: 0, 80: 0, 90: 0}
    _below = {10: 0, 20: 0, 30: 0}

    # ── zero-variance 内部 4 路分解 ──────────────────────────────────────────
    # 仅对 std < zero_var_eps 的 group 分类（其分数可视为均匀常数 = mean）
    # 四类互斥且完备：zv_wrong + zv_type_only + zv_partial + zv_perfect = zero_var_count
    zv_wrong_count = 0        # 均匀分数 < _dist_lo   → 动作类型全错，模型完全无法完成
    zv_type_only_count = 0    # 均匀分数 ∈ [lo, hi]   → 类型对但参数全错，stuck at w_t
    zv_partial_count = 0      # 均匀分数 ∈ (hi, 0.99) → 类型对参数部分对但始终卡在同一分
    zv_perfect_count = 0      # 均匀分数 ≥ 0.99       → 模型已完全饱和

    group_max_scores: list[float] = []
    group_min_scores: list[float] = []
    group_mean_scores: list[float] = []
    group_std_scores: list[float] = []
    group_success_rates: list[float] = []

    # ── rollout-level accumulators ────────────────────────────────────────────
    all_rollout_scores: list[float] = []
    effective_rollout_count = 0
    total_rollout_count = 0

    for scores in uid2scores.values():
        n_r = len(scores)
        total_rollout_count += n_r
        sa = np.array(scores, dtype=np.float64)

        max_s = float(sa.max())
        min_s = float(sa.min())
        mean_s = float(sa.mean())
        std_s = float(sa.std())

        group_max_scores.append(max_s)
        group_min_scores.append(min_s)
        group_mean_scores.append(mean_s)
        group_std_scores.append(std_s)
        group_success_rates.append(float((sa > success_threshold).sum()) / n_r)
        all_rollout_scores.extend(scores)

        # ── zero variance ─────────────────────────────────────────────────────
        if std_s < zero_var_eps:
            zero_var_count += 1
            # 分类到 4 路内部分解：用 mean_s 代表该 group 的均匀分数
            if mean_s < _dist_lo:
                zv_wrong_count += 1
            elif mean_s <= _dist_hi:
                zv_type_only_count += 1
            elif mean_s < 0.99:
                zv_partial_count += 1
            else:
                zv_perfect_count += 1
        else:
            effective_rollout_count += n_r

        # ── backward-compatible all_fail / all_success ────────────────────────
        if max_s <= success_threshold:
            all_fail_count += 1
        if min_s > success_threshold:
            all_success_count += 1

        # ── semantic extremes ─────────────────────────────────────────────────
        if max_s < 0.10:
            all_zero_count += 1     # all scores ≈ 0 (action type totally wrong)
        if min_s >= 0.99:
            all_perfect_count += 1  # all scores = 1.0

        # ── no-params-credit group (weight-aware) ─────────────────────────────
        if max_s <= _dist_hi:
            all_type_only_max_count += 1

        # ── all-above thresholds (min > X/100) ────────────────────────────────
        for t in _above:
            if min_s > t / 100.0:
                _above[t] += 1

        # ── all-below thresholds (max < X/100) ────────────────────────────────
        for t in _below:
            if max_s < t / 100.0:
                _below[t] += 1

    p = total_prompts
    r = max(total_rollout_count, 1)
    all_scores_arr = np.array(all_rollout_scores, dtype=np.float64)

    return {
        # ── backward-compatible basics ─────────────────────────────────────────
        f"{prefix}/total_prompts":          p,
        f"{prefix}/all_fail_count":         all_fail_count,
        f"{prefix}/all_fail_ratio":         all_fail_count / p,
        f"{prefix}/all_success_count":      all_success_count,
        f"{prefix}/all_success_ratio":      all_success_count / p,
        f"{prefix}/any_success_ratio":      1.0 - all_fail_count / p,
        f"{prefix}/avg_success_rate":       float(np.mean(group_success_rates)),
        f"{prefix}/group_max_score/mean":   float(np.mean(group_max_scores)),
        f"{prefix}/group_mean_score/mean":  float(np.mean(group_mean_scores)),

        # ── true GRPO dead zones (zero variance) ──────────────────────────────
        f"{prefix}/zero_variance_count":    zero_var_count,
        f"{prefix}/zero_variance_ratio":    zero_var_count / p,
        f"{prefix}/has_gradient_ratio":     1.0 - zero_var_count / p,

        # ── zero-variance 内部 4 路分解 (over total_prompts) ──────────────────
        # 四类互斥完备：zv_wrong + zv_type_only + zv_partial + zv_perfect = zero_var_count
        #
        # zv_wrong     : 均匀分数 <  lo   (动作类型全错，完全没学会)
        # zv_type_only : 均匀分数 ∈ [lo,hi] (类型对但参数全错，卡在 w_t)
        # zv_partial   : 均匀分数 ∈ (hi,0.99) (类型对参数部分对，卡在同一中间分)
        # zv_perfect   : 均匀分数 ≥ 0.99  (完全正确，已饱和)
        f"{prefix}/zv_wrong_count":         zv_wrong_count,
        f"{prefix}/zv_wrong_ratio":         zv_wrong_count / p,
        f"{prefix}/zv_type_only_count":     zv_type_only_count,
        f"{prefix}/zv_type_only_ratio":     zv_type_only_count / p,
        f"{prefix}/zv_partial_count":       zv_partial_count,
        f"{prefix}/zv_partial_ratio":       zv_partial_count / p,
        f"{prefix}/zv_perfect_count":       zv_perfect_count,
        f"{prefix}/zv_perfect_ratio":       zv_perfect_count / p,

        # ── zero-variance 内部相对比例 (within zero_variance groups) ──────────
        # 用于直接读出"dead zone 里有多少比例是因为全对/全错/卡死"
        **({
            f"{prefix}/zv_composition/all_wrong":    zv_wrong_count      / zero_var_count,
            f"{prefix}/zv_composition/type_only":    zv_type_only_count  / zero_var_count,
            f"{prefix}/zv_composition/partial_stuck": zv_partial_count   / zero_var_count,
            f"{prefix}/zv_composition/all_perfect":  zv_perfect_count    / zero_var_count,
        } if zero_var_count > 0 else {}),

        # ── fine-grained dead-zone semantic breakdown (legacy/cross-check) ────
        f"{prefix}/all_zero_count":               all_zero_count,
        f"{prefix}/all_zero_ratio":               all_zero_count / p,
        f"{prefix}/all_perfect_count":            all_perfect_count,
        f"{prefix}/all_perfect_ratio":            all_perfect_count / p,
        # max_score ≤ action_type_weight*1.1 → no rollout got any params credit
        # semantically equivalent to "all type_wrong or type_only", weight-aware
        f"{prefix}/all_type_only_max_count":      all_type_only_max_count,
        f"{prefix}/all_type_only_max_ratio":      all_type_only_max_count / p,

        # ── all-above threshold breakdowns ────────────────────────────────────
        # min_score > 0.70 / 0.80 / 0.90  (all rollouts in group above threshold)
        # all_above_90_ratio ≡ all_success_ratio when success_threshold=0.9
        f"{prefix}/all_above_70_ratio":     _above[70] / p,
        f"{prefix}/all_above_80_ratio":     _above[80] / p,
        f"{prefix}/all_above_90_ratio":     _above[90] / p,

        # ── all-below threshold breakdowns ────────────────────────────────────
        # max_score < 0.10 / 0.20 / 0.30  (all rollouts in group below threshold)
        f"{prefix}/all_below_10_ratio":     _below[10] / p,
        f"{prefix}/all_below_20_ratio":     _below[20] / p,
        f"{prefix}/all_below_30_ratio":     _below[30] / p,

        # ── group distribution stats ───────────────────────────────────────────
        f"{prefix}/group_min_score/mean":   float(np.mean(group_min_scores)),
        f"{prefix}/group_std/mean":         float(np.mean(group_std_scores)),

        # ── rollout-level effective training-signal ratio ──────────────────────
        # fraction of individual rollouts that belong to a non-zero-variance group
        # (these rollouts may receive a non-zero GRPO advantage)
        f"{prefix}/effective_rollout_ratio": effective_rollout_count / r,

        # ── weight-aware 4-bin score distribution ─────────────────────────────
        # Bin boundaries derived from action_type_weight (w_t):
        #   lo = w_t * 0.5    hi = w_t * 1.1
        # Correct for any weight setting (verified for 0.5/0.5 and 0.2/0.8).
        f"{prefix}/score_dist/type_wrong":  float(np.mean(all_scores_arr < _dist_lo)),
        f"{prefix}/score_dist/type_only":   float(np.mean((all_scores_arr >= _dist_lo) & (all_scores_arr <= _dist_hi))),
        f"{prefix}/score_dist/partial":     float(np.mean((all_scores_arr > _dist_hi)  & (all_scores_arr < 0.99))),
        f"{prefix}/score_dist/perfect":     float(np.mean(all_scores_arr >= 0.99)),
    }


def compute_rollout_group_metrics(
    batch: DataProto,
    n: int,
    success_threshold: float = 0.9,
    prefix: str = "rollout",
    action_type_weight: float = 0.5,
) -> dict[str, Any]:
    """Compute per-prompt group metrics for n rollouts (training).

    Builds a uid → scores mapping from the batch and delegates all computation
    to :func:`_compute_group_stats`.  See that function for the full list of
    returned metrics.

    Args:
        batch: Training batch after rewards have been computed.  Must contain
            ``batch["token_level_scores"]`` and ``non_tensor_batch["uid"]``.
        n: Number of rollouts per prompt (used only for documentation; the actual
            grouping is done by uid).
        success_threshold: Score strictly above this counts as "successful".
        prefix: WandB metric key prefix.
        action_type_weight: Weight for the action-type sub-score (e.g. 0.5 or
            0.2). Controls the bin boundaries of ``score_dist/*`` metrics.
    """
    sequence_scores = batch.batch["token_level_scores"].sum(-1)
    uids = batch.non_tensor_batch["uid"]

    uid2scores: dict[str, list[float]] = defaultdict(list)
    for uid, score in zip(uids, sequence_scores.cpu().tolist()):
        uid2scores[uid].append(score)

    return _compute_group_stats(
        uid2scores,
        prefix=prefix,
        success_threshold=success_threshold,
        action_type_weight=action_type_weight,
    )


def compute_rollout_group_metrics_from_scores(
    scores: np.ndarray,
    n: int,
    success_threshold: float = 0.9,
    prefix: str = "val",
    action_type_weight: float = 0.5,
) -> dict[str, Any]:
    """Compute rollout group metrics from a flat score array (validation).

    Assumes scores are ordered as n consecutive rollouts per prompt:
    ``[r1_p1, r2_p1, …, rn_p1, r1_p2, …, rn_pM]``.
    Converts to a uid → scores dict and delegates to :func:`_compute_group_stats`.

    Args:
        scores: 1-D array of per-rollout scores.
        n: Number of rollouts per prompt.
        success_threshold: Score strictly above this counts as "successful".
        prefix: WandB metric key prefix.
        action_type_weight: Weight for the action-type sub-score (e.g. 0.5 or
            0.2). Controls the bin boundaries of ``score_dist/*`` metrics.
    """
    n_groups = len(scores) // n
    if n_groups == 0:
        return {}

    grouped = scores[: n_groups * n].reshape(n_groups, n)
    uid2scores: dict[str, list[float]] = {
        str(i): grouped[i].tolist() for i in range(n_groups)
    }
    return _compute_group_stats(
        uid2scores,
        prefix=prefix,
        success_threshold=success_threshold,
        action_type_weight=action_type_weight,
    )


# In-domain apps for domain-level aggregation
IN_DOMAIN_APPS = {
    "Audio Recorder",
    "Broccoli Recipe",
    "Camera",
    "Chrome",
    "Clock",
    "Contacts",
    "Files",
    "Joplin",
    "Markor",
    "OpenTracks",
}


def compute_per_group_reward_metrics(
    reward_metrics: dict[str, list[float]],
    group_keys: dict[str, np.ndarray],
    prefix: str = "reward",
    in_domain_apps: set[str] | None = None,
) -> dict[str, Any]:
    """Compute per-app, per-action-type, and domain-level reward metrics.

    Args:
        reward_metrics: Dict of metric_name -> list[float] from reward computation.
            Expected keys: "overall", "action_type", "action_params", "format".
        group_keys: Dict containing grouping arrays:
            - "app_name": np.ndarray of app names per sample
            - "action_type": np.ndarray of action types per sample
        prefix: Metric key prefix (e.g. "reward" or "val").
        in_domain_apps: Set of in-domain app names. If None, uses IN_DOMAIN_APPS.

    Returns:
        Dictionary of per-group metrics ready for WandB logging:
        - ``{prefix}/app/{app_name}/overall``: mean overall score per app
        - ``{prefix}/action/{action_type}/overall``: mean overall score per action type
        - ``{prefix}/domain/in_domain/overall``: aggregated in-domain overall
        - ``{prefix}/domain/ood/overall``: aggregated out-of-domain overall
        - etc.
    """
    metrics = {}

    if in_domain_apps is None:
        in_domain_apps = IN_DOMAIN_APPS

    overall_scores = reward_metrics.get("overall", [])
    action_type_scores = reward_metrics.get("action_type", [])
    action_params_scores = reward_metrics.get("action_params", [])

    if not overall_scores:
        return metrics

    n_samples = len(overall_scores)

    # Per-app metrics + domain aggregation
    app_names = group_keys.get("app_name", None)
    if app_names is not None and len(app_names) == n_samples:
        app2scores = defaultdict(lambda: defaultdict(list))
        # Domain-level aggregation
        domain_scores: dict[str, dict[str, list[float]]] = {
            "in_domain": defaultdict(list),
            "ood": defaultdict(list),
        }

        for i in range(n_samples):
            app = str(app_names[i])
            app2scores[app]["overall"].append(overall_scores[i])
            if i < len(action_type_scores):
                app2scores[app]["action_type"].append(action_type_scores[i])
            if i < len(action_params_scores):
                app2scores[app]["action_params"].append(action_params_scores[i])

            # Classify into in-domain or OOD
            domain_key = "in_domain" if app in in_domain_apps else "ood"
            domain_scores[domain_key]["overall"].append(overall_scores[i])
            if i < len(action_type_scores):
                domain_scores[domain_key]["action_type"].append(action_type_scores[i])
            if i < len(action_params_scores):
                domain_scores[domain_key]["action_params"].append(action_params_scores[i])

        # Per-app metrics
        for app, scores_dict in app2scores.items():
            for metric_name, values in scores_dict.items():
                metrics[f"{prefix}/app/{app}/{metric_name}"] = float(np.mean(values))
            metrics[f"{prefix}/app/{app}/count"] = len(scores_dict["overall"])

        # Domain-level metrics
        for domain_key, scores_dict in domain_scores.items():
            for metric_name, values in scores_dict.items():
                if values:
                    metrics[f"{prefix}/domain/{domain_key}/{metric_name}"] = float(np.mean(values))
            if scores_dict["overall"]:
                metrics[f"{prefix}/domain/{domain_key}/count"] = len(scores_dict["overall"])

    # Per-action-type metrics
    action_types = group_keys.get("action_type", None)
    if action_types is not None and len(action_types) == n_samples:
        action2scores = defaultdict(lambda: defaultdict(list))
        for i in range(n_samples):
            action = str(action_types[i])
            action2scores[action]["overall"].append(overall_scores[i])
            if i < len(action_type_scores):
                action2scores[action]["action_type_score"].append(action_type_scores[i])
            if i < len(action_params_scores):
                action2scores[action]["action_params_score"].append(action_params_scores[i])

        for action, scores_dict in action2scores.items():
            for metric_name, values in scores_dict.items():
                metrics[f"{prefix}/action/{action}/{metric_name}"] = float(np.mean(values))
            metrics[f"{prefix}/action/{action}/count"] = len(scores_dict["overall"])

    return metrics
