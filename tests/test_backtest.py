"""Known-answer tests for backtest statistics."""

import numpy as np
import pytest

from riskplatform.backtest import (
    basel_traffic_light,
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
)


def test_kupiec_known_answer_250_obs_5_breaches():
    # hand-computed: LR = -2[(245 ln .99 + 5 ln .01) - (245 ln .98 + 5 ln .02)]
    lr, p = kupiec_pof(5, 250, alpha=0.99)
    assert lr == pytest.approx(1.9568, abs=1e-3)
    assert 0.15 < p < 0.20


def test_kupiec_exact_rate_gives_zero_lr():
    lr, p = kupiec_pof(25, 500, alpha=0.95)
    assert lr == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0)


def test_kupiec_rejects_gross_overshoot():
    _, p = kupiec_pof(50, 250, alpha=0.99)
    assert p < 1e-6


def test_christoffersen_flags_clustered_breaches():
    breaches = np.array([1] * 10 + [0] * 240)
    _, p = christoffersen_independence(breaches)
    assert p < 0.001


def test_christoffersen_passes_iid_breaches():
    rng = np.random.default_rng(3)
    breaches = (rng.random(500) < 0.05).astype(int)
    _, p = christoffersen_independence(breaches)
    assert p > 0.05


def test_conditional_coverage_combines_both():
    breaches = np.array([1] * 10 + [0] * 240)
    lr_cc, p_cc = conditional_coverage(breaches, alpha=0.99)
    lr_uc, _ = kupiec_pof(10, 250, alpha=0.99)
    lr_ind, _ = christoffersen_independence(breaches)
    assert lr_cc == pytest.approx(lr_uc + lr_ind, rel=1e-9)
    assert p_cc < 0.001


def test_basel_traffic_light_boundaries():
    assert basel_traffic_light(4, 250) == "green"
    assert basel_traffic_light(5, 250) == "yellow"
    assert basel_traffic_light(9, 250) == "yellow"
    assert basel_traffic_light(10, 250) == "red"
    # scaling: 8 breaches over 500 days == 4 per 250 -> green
    assert basel_traffic_light(8, 500) == "green"
