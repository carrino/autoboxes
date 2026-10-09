"""The simple-loony-endgame value (`loony_value`, the solver's leaf) checked three independent
ways: against the bare C++ search on constructed 5x5 endgames with several chains, loops,
1-chains and 2-chains; against BoxesZero's Theorem 5 (how 1-chains and 2-chains change the
value); and against the published controlled-value cases of Theorem 3 (Buzzard & Ciere via
BoxesZero). The paper's v(G) is the controller's margin; ours is the opener's, so v = -ours."""
from __future__ import annotations

import random

import alpha_go_cpp
import pytest

from alpha_go.boxes.chains import components, degrees
from alpha_go.boxes.rules import geometry
from alpha_go.boxes.solver import loony_value

GEO = geometry(5, 5)
FULL = (1 << GEO.num_edges) - 1


def box(r: int, c: int) -> int:
    return r * 5 + c


def link(a: int, b: int) -> int:
    return next(e for e in GEO.box_edges[a] if b in GEO.edge_boxes[e])


def border(b: int, side: str) -> int:
    top, bottom, left, right = GEO.box_edges[b]
    edge = {"top": top, "bottom": bottom, "left": left, "right": right}[side]
    assert len(GEO.edge_boxes[edge]) == 1, "not a border edge"
    return edge


def carve(mask: int, edges: list[int]) -> int:
    for e in edges:
        mask &= ~(1 << e)
    return mask


def chain(mask: int, boxes: list[int], ends: tuple[str, str]) -> int:
    """Undraw a chain from the full mask: its links plus one border edge at each end."""
    links = [link(a, b) for a, b in zip(boxes, boxes[1:])]
    return carve(mask, links + [border(boxes[0], ends[0]), border(boxes[-1], ends[1])])


def loop(mask: int, boxes: list[int]) -> int:
    return carve(mask, [link(a, b) for a, b in zip(boxes, boxes[1:] + boxes[:1])])


def multiset(mask: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    deg = degrees(mask, GEO)
    comps = components(mask, GEO)
    assert max(deg) <= 2 and all(c.is_loop or c.ends == (-1, -1) for c in comps), "not simple"
    return (tuple(sorted(c.size for c in comps if not c.is_loop)),
            tuple(sorted(c.size for c in comps if c.is_loop)))


ROW = [[box(r, c) for c in range(5)] for r in range(5)]
ENDGAMES = {
    "five rows": lambda m: carve(m, [e for e in range(GEO.num_edges)
                                     if e >= (5 + 1) * 5]),  # every vertical edge undrawn
    "5 + 2 + 1 + 4-loop": lambda m: loop(
        chain(chain(chain(m, ROW[0], ("left", "right")), ROW[4][:2], ("bottom", "bottom")),
              [box(4, 4)], ("bottom", "right")),
        [box(1, 1), box(1, 2), box(2, 2), box(2, 1)]),
    "3 + 1 + 3 + 6-loop": lambda m: loop(
        chain(chain(chain(m, ROW[0][:3], ("top", "top")), [box(0, 4)], ("top", "right")),
              [box(2, 0), box(3, 0), box(4, 0)], ("left", "left")),
        [box(1, 2), box(1, 3), box(1, 4), box(2, 4), box(2, 3), box(2, 2)]),
    "3 + 3 + 3": lambda m: chain(
        chain(chain(m, ROW[0][:3], ("top", "top")), ROW[4][2:], ("bottom", "bottom")),
        [box(1, 0), box(2, 0), box(3, 0)], ("left", "left")),
    "two 4-loops + 2": lambda m: chain(
        loop(loop(m, [box(0, 0), box(0, 1), box(1, 1), box(1, 0)]),
             [box(3, 3), box(3, 4), box(4, 4), box(4, 3)]),
        [box(4, 0), box(4, 1)], ("bottom", "bottom")),
    "2 + 2 + 1 + 1": lambda m: chain(
        chain(chain(chain(m, ROW[0][:2], ("top", "top")), ROW[0][3:], ("top", "top")),
              [box(4, 0)], ("bottom", "left")), [box(4, 4)], ("bottom", "right")),
    "2 + 2 + 1 + 1 + 5": lambda m: chain(
        chain(chain(chain(chain(m, ROW[0][:2], ("top", "top")), ROW[0][3:], ("top", "top")),
                    [box(4, 0)], ("bottom", "left")), [box(4, 4)], ("bottom", "right")),
        ROW[2], ("left", "right")),
    "L-chain 9 + 4-loop": lambda m: loop(
        chain(m, ROW[0] + [box(r, 4) for r in range(1, 5)], ("left", "bottom")),
        [box(2, 1), box(2, 2), box(3, 2), box(3, 1)]),
}


class TestConstructedEndgames:
    @pytest.mark.parametrize("name", list(ENDGAMES))
    def test_leaf_equals_bare_search(self, name: str) -> None:
        mask = ENDGAMES[name](FULL)
        chains, loops = multiset(mask)
        with_leaf = alpha_go_cpp.BoxesSolver(5, 5, 1 << 18)
        bare = alpha_go_cpp.BoxesSolver(5, 5, 1 << 20, use_leaf=False, use_equivalence=False)
        assert with_leaf.value(mask) == loony_value(chains, loops), name
        assert bare.value(mask) == loony_value(chains, loops), (name, chains, loops)


def controlled_value(chains: tuple[int, ...], loops: tuple[int, ...]) -> int:
    """c(G) of Theorem 3 (long chains and loops only), with the terminal bonus tb(G)."""
    size = sum(chains) + sum(loops)
    if not chains and not loops:
        bonus = 0
    elif loops and not chains:
        bonus = 8
    elif loops and all(c == 3 for c in chains):
        bonus = 6
    else:
        bonus = 4
    return size - 4 * len(chains) - 8 * len(loops) + bonus


class TestPublishedTheorems:
    def test_theorem_3_case_1_controlled_value(self) -> None:
        # Long components only: whenever c(G) >= 2 the controller wins by exactly c(G).
        rng = random.Random(3)
        checked = 0
        for _ in range(3000):
            chains = tuple(sorted(rng.randint(3, 9) for _ in range(rng.randint(0, 5))))
            loops = tuple(sorted(rng.choice([4, 6, 8, 10]) for _ in range(rng.randint(0, 3))))
            c = controlled_value(chains, loops)
            if c >= 2 and (chains or loops):
                assert -loony_value(chains, loops) == c, (chains, loops, c)
                checked += 1
        assert checked > 500

    def test_theorem_3_case_2_zero_with_a_four_loop(self) -> None:
        rng = random.Random(4)
        checked = 0
        for _ in range(5000):
            chains = tuple(sorted(rng.randint(3, 8) for _ in range(rng.randint(0, 4))))
            loops = tuple(sorted([4] + [rng.choice([4, 6, 8]) for _ in range(rng.randint(0, 2))]))
            if controlled_value(chains, loops) == 0 and (chains, loops) != ((3, 3), (4,)):
                assert loony_value(chains, loops) == 0, (chains, loops)
                checked += 1
        assert checked > 50

    def test_theorem_5_one_and_two_chains(self) -> None:
        # k 1-chains and n 2-chains change the controller's value of the long remainder C as:
        # odd/odd: v(C) - 1; odd/even: 1 - v(C); even/odd: 2 - v(C); even/even: v(C).
        rng = random.Random(5)
        for _ in range(2000):
            long_chains = tuple(sorted(rng.randint(3, 8) for _ in range(rng.randint(0, 4))))
            loops = tuple(sorted(rng.choice([4, 6, 8]) for _ in range(rng.randint(0, 2))))
            k, n = rng.randint(0, 4), rng.randint(0, 4)
            if not long_chains and not loops:
                continue
            v_c = -loony_value(long_chains, loops)
            rule = {(1, 1): v_c - 1, (1, 0): 1 - v_c, (0, 1): 2 - v_c, (0, 0): v_c}
            expected = rule[(k % 2, n % 2)]
            chains = tuple(sorted(long_chains + (1,) * k + (2,) * n))
            assert -loony_value(chains, loops) == expected, (chains, loops, k, n, v_c)
