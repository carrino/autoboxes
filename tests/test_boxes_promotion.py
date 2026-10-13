"""Promotion counts game wins and respects correlation within opening pairs."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from alpha_go.boxes.arena import MatchResult


def match(margins: list[int]) -> MatchResult:
    return MatchResult("a", "b", 5, 5, len(margins), sum(m > 0 for m in margins),
                       sum(m < 0 for m in margins), sum(m == 0 for m in margins),
                       margins, 0, 0)


def test_game_wins_can_promote_despite_losing_box_margin_pairs() -> None:
    result = match([1, -9] * 60 + [1, 1] * 40)
    assert result.a_win_rate == 0.7 and result.pair_score == 0.4
    assert result.wins_promotion()
    assert not match([-x for x in result.margins]).wins_promotion()


def test_split_pairs_never_imply_improvement_and_tiny_sample_cannot_promote() -> None:
    assert match([9, -1] * 100).paired_win_pvalue() == 1
    assert not match([9, -1] * 100).wins_promotion()
    assert not match([1, 1] * 4).wins_promotion()
    assert match([1, 1] * 6).paired_win_pvalue() == pytest.approx(2 / 64)


def test_pairing_changes_evidence_even_when_game_totals_match() -> None:
    correlated = match([1, 1] * 7 + [-1, -1] * 3)
    mixed = match([1, -1] * 6 + [1, 1] * 4)
    assert correlated.a_win_rate == mixed.a_win_rate == 0.7
    assert correlated.paired_win_pvalue() > mixed.paired_win_pvalue()


def test_dynamic_test_matches_enumerated_swaps_with_draws() -> None:
    margins = [1, 0, 1, 1, -1, 0, 1, -1, 1, 1]
    pairs = np.sign(margins).reshape(-1, 2).sum(axis=1)
    observed = abs(pairs.sum())
    null = [abs(sum(p * s for p, s in zip(pairs, signs)))
            for signs in itertools.product((-1, 1), repeat=len(pairs))]
    expected = sum(s >= observed for s in null) / len(null)
    assert match(margins).paired_win_pvalue() == pytest.approx(expected)
