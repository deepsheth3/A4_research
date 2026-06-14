"""CPU tests for speculative-decoding acceptance metrics (acceptance.py)."""

import math

import torch

from turboquant.validation.acceptance import (
    acceptance_stats, expected_accepted_length, speedup,
)


def test_identical_distributions_accept_fully():
    logits = torch.randn(50, 30)
    alpha, greedy = acceptance_stats(logits, logits)
    assert abs(alpha - 1.0) < 1e-5
    assert abs(greedy - 1.0) < 1e-5


def test_different_distributions_accept_less():
    d = torch.randn(50, 30)
    t = torch.randn(50, 30)
    alpha, greedy = acceptance_stats(d, t)
    assert 0.0 <= alpha < 1.0
    assert 0.0 <= greedy <= 1.0


def test_closer_draft_has_higher_alpha():
    t = torch.randn(200, 40)
    near = t + 0.1 * torch.randn_like(t)
    far = t + 2.0 * torch.randn_like(t)
    a_near, _ = acceptance_stats(near, t)
    a_far, _ = acceptance_stats(far, t)
    assert a_near > a_far


def test_expected_accepted_length_formula():
    # closed form: (1 - a^(g+1))/(1-a)
    a, g = 0.8, 4
    assert math.isclose(expected_accepted_length(a, g),
                        (1 - a ** (g + 1)) / (1 - a), rel_tol=1e-9)
    assert expected_accepted_length(1.0, 4) == 5.0     # alpha=1 -> gamma+1
    # monotone increasing in alpha
    assert expected_accepted_length(0.9, 4) > expected_accepted_length(0.5, 4)


def test_speedup_monotone_in_alpha():
    s_lo = speedup(0.5, 4, 0.1)
    s_hi = speedup(0.9, 4, 0.1)
    assert s_hi > s_lo > 0
    # a fast, accurate draft beats plain decoding
    assert speedup(0.9, 4, 0.05) > 1.0
