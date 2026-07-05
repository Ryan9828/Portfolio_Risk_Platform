"""Typed settings loaded from config/portfolio.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "portfolio.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

PORTFOLIO_TICKER = "PORTFOLIO"  # pseudo-ticker for the weighted portfolio series


@dataclass(frozen=True)
class GarchConfig:
    default: str = "GARCH"
    egarch_assets: tuple[str, ...] = ()
    dist: str = "t"
    rescale: float = 100.0


@dataclass(frozen=True)
class VarConfig:
    confidences: tuple[float, ...] = (0.95, 0.99)
    horizons: tuple[int, ...] = (1, 10)
    mc_sims: int = 10_000
    mc_seed: int = 42
    hs_window: int = 500


@dataclass(frozen=True)
class BacktestConfig:
    window: int = 500
    refit_every: int = 5


@dataclass(frozen=True)
class MonitoringConfig:
    psi_bins: int = 10
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    psi_reference_window: tuple[int, int] = (750, 250)
    psi_current_window: int = 60
    jump_zscore: float = 6.0
    jump_vol_window: int = 60
    stale_days_alert: int = 3


@dataclass(frozen=True)
class IntelConfig:
    """Announcement-intelligence stage (LLM extraction over ASX announcements)."""

    model: str = "claude-opus-4-8"
    max_new_per_run: int = 25          # cost guard: LLM calls per pipeline run
    per_ticker_fetch: int = 20         # announcements pulled per ticker per run
    input_usd_per_mtok: float = 5.0    # for the cost column on each signal row
    output_usd_per_mtok: float = 25.0
    event_vol_window: int = 20         # sessions either side for realised-vol comparison
    event_reaction_days: int = 3       # window to look for a post-announcement jump
    event_jump_zscore: float = 2.0     # |z| threshold that counts as a reaction


@dataclass(frozen=True)
class Settings:
    weights: dict[str, float]
    benchmarks: tuple[str, ...]
    index_ticker: str
    backfill_years: int
    min_obs_garch: int
    max_ffill_days: int
    ingest_overlap_days: int
    garch: GarchConfig = field(default_factory=GarchConfig)
    var: VarConfig = field(default_factory=VarConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    intel: IntelConfig = field(default_factory=IntelConfig)

    @property
    def portfolio_tickers(self) -> list[str]:
        return list(self.weights)

    @property
    def all_tickers(self) -> list[str]:
        return list(self.weights) + [t for t in self.benchmarks if t not in self.weights]


def load_settings(path: Path | str = DEFAULT_CONFIG) -> Settings:
    raw = yaml.safe_load(Path(path).read_text())

    assets = raw["assets"]
    weights = {str(k): float(v) for k, v in assets["portfolio"].items()}
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Portfolio weights sum to {total:.6f}, expected 1.0")

    hist = raw["history"]
    g = raw["models"]["garch"]
    v = raw["models"]["var"]
    bt = raw["backtest"]
    m = raw["monitoring"]
    intel = raw.get("intel", {})

    return Settings(
        weights=weights,
        benchmarks=tuple(assets.get("benchmarks", [])),
        index_ticker=assets["index_ticker"],
        backfill_years=int(hist["backfill_years"]),
        min_obs_garch=int(hist["min_obs_garch"]),
        max_ffill_days=int(hist["max_ffill_days"]),
        ingest_overlap_days=int(hist["ingest_overlap_days"]),
        garch=GarchConfig(
            default=g["default"],
            egarch_assets=tuple(g.get("egarch_assets", [])),
            dist=g["dist"],
            rescale=float(g["rescale"]),
        ),
        var=VarConfig(
            confidences=tuple(float(c) for c in v["confidences"]),
            horizons=tuple(int(h) for h in v["horizons"]),
            mc_sims=int(v["mc_sims"]),
            mc_seed=int(v["mc_seed"]),
            hs_window=int(v["hs_window"]),
        ),
        backtest=BacktestConfig(window=int(bt["window"]), refit_every=int(bt["refit_every"])),
        monitoring=MonitoringConfig(
            psi_bins=int(m["psi_bins"]),
            psi_warn=float(m["psi_warn"]),
            psi_alert=float(m["psi_alert"]),
            psi_reference_window=tuple(int(x) for x in m["psi_reference_window"]),
            psi_current_window=int(m["psi_current_window"]),
            jump_zscore=float(m["jump_zscore"]),
            jump_vol_window=int(m["jump_vol_window"]),
            stale_days_alert=int(m["stale_days_alert"]),
        ),
        intel=IntelConfig(
            model=str(intel.get("model", IntelConfig.model)),
            max_new_per_run=int(intel.get("max_new_per_run", IntelConfig.max_new_per_run)),
            per_ticker_fetch=int(intel.get("per_ticker_fetch", IntelConfig.per_ticker_fetch)),
            input_usd_per_mtok=float(intel.get("input_usd_per_mtok", IntelConfig.input_usd_per_mtok)),
            output_usd_per_mtok=float(intel.get("output_usd_per_mtok", IntelConfig.output_usd_per_mtok)),
            event_vol_window=int(intel.get("event_vol_window", IntelConfig.event_vol_window)),
            event_reaction_days=int(intel.get("event_reaction_days", IntelConfig.event_reaction_days)),
            event_jump_zscore=float(intel.get("event_jump_zscore", IntelConfig.event_jump_zscore)),
        ),
    )
