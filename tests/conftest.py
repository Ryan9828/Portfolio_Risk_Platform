"""Shared fixtures: deterministic synthetic market data so CI never touches the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskplatform.config import load_settings

N_DAYS = 800
SEED = 7


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def fixture_prices(settings) -> pd.DataFrame:
    """GBM price paths for every configured ticker on a business-day calendar."""
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(end="2026-06-30", periods=N_DAYS)
    vols = {t: 0.012 for t in settings.all_tickers}
    vols["AUDUSD=X"] = 0.006
    frames = []
    for ticker, vol in vols.items():
        rets = rng.standard_normal(N_DAYS) * vol + 0.0002
        prices = 100 * np.exp(np.cumsum(rets))
        frames.append(pd.DataFrame({"date": dates, "ticker": ticker, "adj_close": prices}))
    return pd.concat(frames, ignore_index=True)
