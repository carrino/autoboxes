"""Relative full-game strength against a fixed external-engine configuration.

Game score, not summed box-margin pair wins, defines the Elo difference. Opening
pairs are the independent sampling units. This is a conditional match rating,
not an absolute or published engine Elo.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from alpha_go.boxes.arena import MatchResult


def _elo(score: float) -> float | str:
    """Retain unbounded endpoint estimates without invalid JSON infinities."""
    if score == 0:
        return "-infinity"
    if score == 1:
        return "infinity"
    return 400 * math.log10(score / (1 - score))


def summarize_margins(margins: list[int], seed: int) -> dict[str, Any]:
    """Summarize a fixed complete match, adjacent games sharing an opening.

    Positive margins are candidate wins; zero counts half. Even games have the
    candidate first. Percentile bootstrap draws retain both sides of each pair.
    A universally reported Hoeffding bound also supplies a nondegenerate interval
    when the observed pair bootstrap has no spread. Its guarantee assumes
    independent opening pairs; neither interval corrects sequential stopping.
    """
    if not margins or len(margins) % 2 or any(type(m) is not int for m in margins):
        raise ValueError("Require a positive even number of integer game margins")
    if type(seed) is not int or seed < 0:
        raise ValueError("Require a nonnegative integer bootstrap seed")
    scores = np.array([(m > 0) + .5 * (m == 0) for m in margins], dtype=np.float64)
    pairs = scores.reshape(-1, 2).mean(axis=1)
    score = float(scores.mean())
    resamples = 10000
    indices = np.random.default_rng(seed).integers(0, len(pairs), (resamples, len(pairs)))
    bootstrap = np.quantile(pairs[indices].mean(axis=1), [.025, .975]).tolist()
    radius = math.sqrt(math.log(2 / .05) / (2 * len(pairs)))
    hoeffding = [max(0., score - radius), min(1., score + radius)]
    degenerate = bootstrap[0] == bootstrap[1]
    interval = hoeffding if degenerate else bootstrap
    wins, losses, draws = (sum(m > 0 for m in margins), sum(m < 0 for m in margins),
                           sum(m == 0 for m in margins))
    match = MatchResult("candidate", "opponent", 5, 5, len(margins), wins, losses,
                        draws, margins, 0., 0.)
    return dict(
        games=len(margins), opening_pairs=len(pairs), wins=wins, losses=losses, draws=draws,
        game_score=score, first_score=float(scores[::2].mean()),
        second_score=float(scores[1::2].mean()), mean_margin=float(np.mean(margins)),
        paired_win_pvalue=match.paired_win_pvalue(),
        pair_bootstrap_ci95=bootstrap, hoeffding_ci95=hoeffding,
        game_score_ci95=interval,
        interval_method=("Hoeffding bound (degenerate pair bootstrap)" if degenerate
                         else "opening-pair cluster percentile bootstrap"),
        relative_elo=_elo(score), relative_elo_ci95=[_elo(endpoint) for endpoint in interval],
        bootstrap_seed=seed, bootstrap_resamples=resamples,
        scope="Relative to this opponent and match configuration, not published absolute Elo. "
              "Infinity strings denote unbounded logit estimates, not proven infinite strength. "
              "Fixed complete match; independent opening pairs; no sequential-stop correction.",
    )
