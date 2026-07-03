"""Monitoring checks on synthetic frames."""

import numpy as np
import pandas as pd

from riskplatform.config import MonitoringConfig
from riskplatform.monitoring import (
    check_extreme_jumps,
    check_stale_prices,
    compute_psi,
)

CFG = MonitoringConfig()


def test_psi_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    assert compute_psi(x, x) < 1e-9


def test_psi_shifted_distribution_alerts():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(1000)
    cur = rng.standard_normal(500) + 2.0
    assert compute_psi(cur, ref) > 0.25


def test_stale_price_detector():
    dates = pd.bdate_range("2026-01-01", periods=100)
    live = np.linspace(100, 110, 100)
    stale = live.copy()
    stale[-5:] = stale[-6]  # frozen for the last 5 sessions
    frame = pd.DataFrame({"LIVE": live, "STALE": stale}, index=dates)
    result = check_stale_prices(frame, CFG)
    assert result.status == "ALERT"
    assert "STALE" in result.detail and "LIVE" not in result.detail


def test_extreme_jump_detector():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2026-01-01", periods=200)
    calm = rng.standard_normal(200) * 0.01
    jumpy = calm.copy()
    jumpy[-1] = 0.30  # 30% move on ~1% vol
    frame = pd.DataFrame({"CALM": calm, "JUMPY": jumpy}, index=dates)
    result = check_extreme_jumps(frame, CFG)
    assert result.status == "ALERT"
    assert "JUMPY" in result.detail and "CALM" not in result.detail
