"""External ratings use game results, retain paired uncertainty and honest bounds."""
from __future__ import annotations

import json
import math

import pytest

from alpha_go.boxes.external_rating import summarize_margins


def test_rating_uses_game_score_not_box_margin_pair_wins() -> None:
    result = summarize_margins([1, -9] * 60 + [1, 1] * 40, seed=7)
    assert result["game_score"] == .7
    assert result["mean_margin"] < 0
    assert result["relative_elo"] == pytest.approx(400 * math.log10(.7 / .3))
    assert result["first_score"] == 1 and result["second_score"] == .4
    assert "cluster" in result["interval_method"]


def test_pairing_changes_uncertainty_with_identical_game_totals() -> None:
    correlated = summarize_margins([1, 1] * 70 + [-1, -1] * 30, seed=12)
    mixed = summarize_margins([1, -1] * 60 + [1, 1] * 40, seed=12)
    assert correlated["game_score"] == mixed["game_score"] == .7
    assert correlated["pair_bootstrap_ci95"][0] < mixed["pair_bootstrap_ci95"][0]
    assert correlated["pair_bootstrap_ci95"][1] > mixed["pair_bootstrap_ci95"][1]
    assert correlated["paired_win_pvalue"] > mixed["paired_win_pvalue"]


def test_draws_half_scores_and_reversing_players() -> None:
    margins = [1, 0] * 20 + [1, 1] * 10 + [-1, 0] * 10
    a = summarize_margins(margins, seed=8)
    b = summarize_margins([-m for m in margins], seed=8)
    assert (a["wins"], a["losses"], a["draws"]) == (40, 10, 30)
    assert a["game_score"] == (40 + 15) / 80
    assert b["relative_elo"] == pytest.approx(-a["relative_elo"])
    assert b["relative_elo_ci95"] == pytest.approx([-x for x in a["relative_elo_ci95"][::-1]])
    assert b["paired_win_pvalue"] == pytest.approx(a["paired_win_pvalue"])
    assert a == summarize_margins(margins, seed=8)


@pytest.mark.parametrize("sign,endpoint", [(1, "infinity"), (-1, "-infinity")])
def test_unanimous_match_has_conservative_finite_other_bound_and_strict_json(
    sign: int, endpoint: str,
) -> None:
    result = summarize_margins([sign] * 100, seed=9)
    assert result["relative_elo"] == endpoint
    assert "Hoeffding" in result["interval_method"]
    assert result["game_score_ci95"] == result["hoeffding_ci95"]
    finite = result["relative_elo_ci95"][0 if sign > 0 else 1]
    assert finite == pytest.approx(sign * 249.571771, abs=.001)
    assert result["relative_elo_ci95"][1 if sign > 0 else 0] == endpoint
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("margins", [[1, -1] * 50, [0, 0] * 50])
def test_identical_half_score_pairs_do_not_claim_zero_uncertainty(margins: list[int]) -> None:
    result = summarize_margins(margins, seed=3)
    assert result["relative_elo"] == 0
    assert result["pair_bootstrap_ci95"] == [.5, .5]
    assert result["game_score_ci95"][0] < .5 < result["game_score_ci95"][1]
    assert result["paired_win_pvalue"] == 1


@pytest.mark.parametrize("margins", [[], [1], [1, 2, 3], [1., -1.], [True, False]])
def test_invalid_match_rejected(margins: list[int]) -> None:
    with pytest.raises(ValueError, match="positive even"):
        summarize_margins(margins, seed=0)


def test_invalid_seed_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        summarize_margins([1, -1], seed=-1)
