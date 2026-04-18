"""
backtest.py
-----------
Realistic backtesting engine with:
  - transaction costs (0.1 %) and slippage (0.05 %)
  - full position sizing
  - ROI, Sharpe Ratio, Max Drawdown, Annualised Return
  - equity curve & drawdown visualisation
  - buy-and-hold benchmark comparison
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import CLASS_NAMES, ensure_dir, get_logger

logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  Core engine
# ═════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Event-driven daily backtester.

    Signal convention (matches paper labelling):
        1 → BUY   (open long position)
        0 → SELL  (close long / open short)
        2 → HOLD  (maintain current position)

    Assumptions
    -----------
    - One asset, long-only (no short selling by default).
    - Position sizing: `position_size_pct` of current equity per trade.
    - Execution at next-day open (to avoid look-ahead bias).
    - Transaction cost and slippage applied symmetrically on open/close.
    """

    def __init__(
        self,
        initial_capital:   float = 100_000.0,
        transaction_cost:  float = 0.001,     # 0.1 %
        slippage:          float = 0.0005,    # 0.05 %
        position_size_pct: float = 1.0,
        risk_free_rate:    float = 0.04,
        allow_short:       bool  = False,
    ):
        self.initial_capital   = initial_capital
        self.tc                = transaction_cost
        self.slippage          = slippage
        self.pos_size          = position_size_pct
        self.risk_free_rate    = risk_free_rate
        self.allow_short       = allow_short

    # ── run ──────────────────────────────────────────────────────────────────

    def run(
        self,
        prices:  pd.Series,    # DatetimeIndex, Close prices
        signals: np.ndarray,   # 0=SELL, 1=BUY, 2=HOLD  (aligned with prices)
    ) -> Dict:
        """
        Execute the backtest; return a results dict containing the equity curve
        and all performance metrics.
        """
        assert len(prices) == len(signals), "prices and signals must be the same length"

        capital   = self.initial_capital
        position  = 0.0      # shares held
        equity    = [capital]
        trades: List[Dict] = []
        in_position = False
        entry_price = 0.0

        for i in range(len(prices) - 1):
            sig  = signals[i]
            exec_price = float(prices.iloc[i + 1])  # next-day open (approx. close)

            # ── OPEN LONG ────────────────────────────────────────────────────
            if sig == 1 and not in_position:
                effective_price = exec_price * (1 + self.slippage)
                cost_per_share  = effective_price * (1 + self.tc)
                affordable      = capital * self.pos_size
                position        = affordable / cost_per_share
                capital        -= position * cost_per_share
                entry_price     = effective_price
                in_position     = True
                trades.append({"date": prices.index[i + 1], "action": "BUY",
                                "price": effective_price, "shares": position})

            # ── CLOSE LONG ───────────────────────────────────────────────────
            elif sig == 0 and in_position:
                effective_price = exec_price * (1 - self.slippage)
                proceeds        = position * effective_price * (1 - self.tc)
                capital        += proceeds
                trades.append({"date": prices.index[i + 1], "action": "SELL",
                                "price": effective_price, "shares": position,
                                "pnl": proceeds - position * entry_price})
                position    = 0.0
                in_position = False

            # ── Mark-to-market equity ─────────────────────────────────────────
            mark = capital + position * float(prices.iloc[i + 1])
            equity.append(mark)

        # Close any open position at end
        if in_position and len(prices) > 0:
            final_price = float(prices.iloc[-1]) * (1 - self.slippage)
            proceeds    = position * final_price * (1 - self.tc)
            capital    += proceeds
            equity[-1]  = capital

        equity_series = pd.Series(equity, index=prices.index, name="equity")
        return {
            "equity":    equity_series,
            "trades":    pd.DataFrame(trades) if trades else pd.DataFrame(),
            "metrics":   self._compute_metrics(equity_series),
        }

    # ── metrics ──────────────────────────────────────────────────────────────

    def _compute_metrics(self, equity: pd.Series) -> Dict[str, float]:
        returns = equity.pct_change().dropna()
        n_days  = len(equity)

        final_val      = equity.iloc[-1]
        total_ret      = (final_val - self.initial_capital) / self.initial_capital
        ann_ret        = (final_val / self.initial_capital) ** (252 / max(n_days, 1)) - 1
        daily_rf       = (1 + self.risk_free_rate) ** (1 / 252) - 1
        excess_returns = returns - daily_rf
        sharpe         = (
            excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            if excess_returns.std() > 0 else 0.0
        )
        peak    = equity.cummax()
        dd      = (equity - peak) / peak
        max_dd  = float(dd.min())
        n_trades = 0 if not hasattr(self, "_trades") else len(self._trades)

        return {
            "roi":               float(total_ret),
            "annualised_return": float(ann_ret),
            "sharpe_ratio":      float(sharpe),
            "max_drawdown":      float(max_dd),
            "final_value":       float(final_val),
        }


# ═════════════════════════════════════════════════════════════════════════════
#  Buy-and-Hold benchmark
# ═════════════════════════════════════════════════════════════════════════════

def buy_and_hold(
    prices:          pd.Series,
    initial_capital: float = 100_000.0,
    tc:              float = 0.001,
) -> Dict:
    """Simple buy-and-hold: buy on day 1, sell on last day."""
    entry = float(prices.iloc[0]) * (1 + tc)
    shares = (initial_capital * (1 - tc)) / entry
    equity = pd.Series(shares * prices.values, index=prices.index)
    engine = BacktestEngine(initial_capital=initial_capital)
    dummy_signals = np.full(len(prices), 2)  # all HOLD
    metrics = engine._compute_metrics(equity)
    return {"equity": equity, "metrics": metrics}


# ═════════════════════════════════════════════════════════════════════════════
#  Visualisations
# ═════════════════════════════════════════════════════════════════════════════

def plot_equity_curve(
    strategy_equity:   pd.Series,
    bh_equity:         pd.Series,
    prices:            pd.Series,
    signals:           Optional[np.ndarray] = None,
    ticker:            str = "",
    save_path:         Optional[str | Path] = None,
) -> plt.Figure:
    """
    Two-panel plot:
      Top  — price chart with buy/sell signal markers
      Bottom — equity curve vs buy-and-hold + drawdown fill
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2, 1]})

    # ── Panel 1: price with signals ──────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(prices.index, prices.values, color="#2c3e50", lw=1.2, label="Price")
    if signals is not None:
        buy_mask  = signals == 1
        sell_mask = signals == 0
        if buy_mask.any():
            ax1.scatter(prices.index[buy_mask],  prices.values[buy_mask],
                        marker="^", color="#2ecc71", s=60, zorder=5, label="BUY")
        if sell_mask.any():
            ax1.scatter(prices.index[sell_mask], prices.values[sell_mask],
                        marker="v", color="#e74c3c", s=60, zorder=5, label="SELL")
    ax1.set_title(f"{ticker} — Price with Signals", fontsize=12)
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")

    # ── Panel 2: equity curves ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(strategy_equity.index, strategy_equity.values,
             color="#e67e22", lw=1.5, label="Strategy")
    ax2.plot(bh_equity.index,       bh_equity.values,
             color="#3498db", lw=1.5, linestyle="--", label="Buy & Hold")
    ax2.set_title("Equity Curve", fontsize=12)
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.legend()

    # ── Panel 3: drawdown ────────────────────────────────────────────────────
    ax3 = axes[2]
    peak = strategy_equity.cummax()
    dd   = (strategy_equity - peak) / peak * 100
    ax3.fill_between(dd.index, dd.values, 0, color="#e74c3c", alpha=0.5)
    ax3.set_title("Drawdown (%)", fontsize=12)
    ax3.set_ylabel("DD (%)")

    fig.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info(f"Equity curve saved → {save_path}")
    return fig


def plot_backtest_comparison(
    strategy_metrics: Dict[str, float],
    bh_metrics:       Dict[str, float],
    save_path:        Optional[str | Path] = None,
) -> plt.Figure:
    """Bar chart comparing strategy vs buy-and-hold metrics."""
    keys   = ["roi", "annualised_return", "sharpe_ratio", "max_drawdown"]
    labels = ["ROI", "Ann. Return", "Sharpe Ratio", "Max Drawdown"]
    strat_v = [strategy_metrics.get(k, 0) for k in keys]
    bh_v    = [bh_metrics.get(k, 0)       for k in keys]

    x = np.arange(len(keys))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, strat_v, w, label="Strategy",     color="#e67e22")
    b2 = ax.bar(x + w/2, bh_v,    w, label="Buy & Hold",   color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title("Strategy vs Buy-and-Hold", fontsize=13)
    ax.legend()
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  Convenience wrapper
# ═════════════════════════════════════════════════════════════════════════════

def run_full_backtest(
    prices:   pd.Series,
    signals:  np.ndarray,
    cfg:      dict,
    ticker:   str = "",
    save_dir: Optional[str | Path] = None,
) -> Dict:
    """
    Run strategy backtest + buy-and-hold; return combined results dict.
    """
    b_cfg = cfg.get("backtest", {})
    engine = BacktestEngine(
        initial_capital=b_cfg.get("initial_capital", 100_000),
        transaction_cost=b_cfg.get("transaction_cost", 0.001),
        slippage=b_cfg.get("slippage", 0.0005),
        position_size_pct=b_cfg.get("position_size_pct", 1.0),
        risk_free_rate=b_cfg.get("risk_free_rate", 0.04),
    )
    strategy = engine.run(prices, signals)
    bh       = buy_and_hold(prices, initial_capital=b_cfg.get("initial_capital", 100_000))

    logger.info(f"\n{'='*50}")
    logger.info(f"  {ticker} Backtest Results")
    logger.info(f"{'='*50}")
    for k, v in strategy["metrics"].items():
        logger.info(f"  Strategy  {k:>20} : {v:.4f}")
    for k, v in bh["metrics"].items():
        logger.info(f"  B&H       {k:>20} : {v:.4f}")

    if save_dir:
        plot_equity_curve(
            strategy["equity"], bh["equity"], prices, signals,
            ticker=ticker,
            save_path=Path(save_dir) / f"{ticker}_equity_curve.png",
        )
        plot_backtest_comparison(
            strategy["metrics"], bh["metrics"],
            save_path=Path(save_dir) / f"{ticker}_backtest_comparison.png",
        )

    return {
        "strategy": strategy,
        "buy_and_hold": bh,
    }
