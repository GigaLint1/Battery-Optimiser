"""
Level 1: Linear Programming (Perfect Foresight) Battery Optimizer.

This module solves the deterministic LP: given known future prices for all
time periods, find the charge/discharge schedule that maximizes revenue.

This is the foundational model. Every subsequent level (DP, stochastic
programming, MPC, RL) is a response to a specific limitation of this model.
Understanding *why* this LP is both powerful and limited is the goal.

Mathematical formulation
------------------------
Decision variables:
    c_t  ≥ 0   : charge power at time t (MW)
    d_t  ≥ 0   : discharge power at time t (MW)
    s_t  ≥ 0   : state of charge at time t (MWh)

Objective (maximize):
    Σ_t [ price_t * d_t  -  price_t * c_t  -  degradation * (c_t + d_t) ] * Δt

Where Δt is the time step length in hours (1.0 for hourly data).

Constraints:
    [SOC dynamics]  s_{t+1} = s_t + η_c * c_t * Δt - (1/η_d) * d_t * Δt
    [Power limits]  0 ≤ c_t ≤ P_max
                    0 ≤ d_t ≤ P_max
    [SOC limits]    SOC_min ≤ s_t ≤ SOC_max   for all t
    [Initial SOC]   s_0 = SOC_initial

IMPORTANT NOTE on simultaneous charging and discharging:
    There is no explicit constraint preventing c_t > 0 AND d_t > 0 at the
    same time. This is intentional. When prices are positive, the LP
    objective penalizes any energy that is "cycled" for no net gain — so
    it naturally sets one of them to zero. This breaks under negative
    prices! Study the challenge question in the learning pathway.

Dependencies: cvxpy, numpy, pandas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp
from dataclasses import dataclass, field
from typing import Optional

from .battery import BatteryConfig


@dataclass
class LPResult:
    """
    Results from an LP optimization run.

    Keeping results in a structured dataclass (rather than raw arrays)
    makes it easy to pass results between functions, log them, and
    compare across methods (LP vs DP vs MPC).
    """
    # Optimization status
    status: str                        # "optimal", "infeasible", etc.
    total_revenue: float               # $ total over the horizon

    # Time series (indexed to match the input price series)
    charge_mw: np.ndarray              # MW charged each period
    discharge_mw: np.ndarray          # MW discharged each period
    soc_mwh: np.ndarray               # SOC at start of each period (MWh)

    # Dual variables (shadow prices) — economic insight lives here
    dual_soc: Optional[np.ndarray] = None  # Shadow price on SOC constraint

    # Summary metrics
    n_periods: int = 0
    n_cycles: float = 0.0             # Total energy cycled / usable capacity
    revenue_per_kw_yr: float = 0.0    # Annualized revenue per kW of capacity

    def to_dataframe(self, timestamps=None) -> pd.DataFrame:
        """Convert results to a DataFrame for plotting and analysis."""
        df = pd.DataFrame({
            'charge_mw': self.charge_mw,
            'discharge_mw': self.discharge_mw,
            'soc_mwh': self.soc_mwh,
            'net_power_mw': self.discharge_mw - self.charge_mw,
        })
        if timestamps is not None:
            df.index = timestamps
        return df


class LPOptimizer:
    """
    Perfect-foresight LP optimizer for battery dispatch.

    Usage:
        battery = BatteryConfig()
        optimizer = LPOptimizer(battery)
        result = optimizer.solve(prices_array)

    The solve() method is the entry point. It formulates and solves the LP,
    then packages results into an LPResult object.
    """

    def __init__(self, battery: BatteryConfig, dt_hours: float = 1.0):
        """
        Args:
            battery: Physical parameters of the battery.
            dt_hours: Length of each time step in hours (1.0 for hourly data,
                      0.25 for 15-minute intervals, etc.)
        """
        self.battery = battery
        self.dt = dt_hours

    def solve(self, prices: np.ndarray) -> LPResult:
        """
        Solve the perfect-foresight LP for the given price series.

        Args:
            prices: Array of electricity prices ($/MWh), one per time period.
                    Length determines the optimization horizon.

        Returns:
            LPResult with optimal schedule and diagnostics.

        Raises:
            ValueError: If prices contain NaN or the problem is infeasible.
        """
        prices = np.asarray(prices, dtype=float)
        if np.any(np.isnan(prices)):
            raise ValueError("Price array contains NaN values.")

        T = len(prices)
        bat = self.battery
        dt = self.dt

        # ── Decision Variables ──────────────────────────────────────────────
        # c = charge (MW)
        # d = discharge (MW)
        # s = state of charge (MWh)
        c = cp.Variable(T)
        d = cp.Variable(T)
        s = cp.Variable(T+1)

        # ── Objective ───────────────────────────────────────────────────────
        objective = cp.Maximize(
            dt * cp.sum(cp.multiply(prices,d-c) - bat.degradation_cost * (c + d))
        )

        # ── Constraints ─────────────────────────────────────────────────────
        constraints = [
            # Initial SOC constraint
            s[0] == bat.soc_initial_mwh,

            # 48 SOC dynamics 
            s[1:] == s[:-1] + bat.rte_charge * c * dt - (1/bat.rte_discharge) * d * dt,

            # Power limits
            c <= bat.power_mw,
            c >= 0,
            d <= bat.power_mw,
            d >= 0,
            
            # Capacity limits
            s >= bat.soc_min_mwh,
            s <= bat.soc_max_mwh

        ]

        # ── Solve ────────────────────────────────────────────────────────────
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.HIGHS)

        # ── Extract Results ──────────────────────────────────────────────────
        
        # For infeasible or unbounded results
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return LPResult(
                status=prob.status,
                total_revenue=0.0,
                charge_mw=np.zeros(T),
                discharge_mw=np.zeros(T),
                soc_mwh=np.zeros(T + 1),
            )
        
        # For Optimal 
        charge_mw    = c.value
        discharge_mw = d.value
        soc_mwh      = s.value
        dual_soc     = constraints[1].dual_value   # SOC dynamics is index 1

        total_revenue       = float(prob.value)
        n_cycles            = self._compute_cycles(charge_mw)
        revenue_per_kw_yr   = self._annualize_revenue(total_revenue, T)

        return LPResult(
            status=prob.status,
            total_revenue=total_revenue,
            charge_mw=charge_mw,
            discharge_mw=discharge_mw,
            soc_mwh=soc_mwh,
            dual_soc=dual_soc,
            n_periods=T,
            n_cycles=n_cycles,
            revenue_per_kw_yr=revenue_per_kw_yr,
        )

    def _compute_cycles(self, charge_mw: np.ndarray) -> float:
        """
        Compute the number of equivalent full cycles.

        A "cycle" is defined as charging the full usable capacity once.
        Total energy charged / usable capacity = number of cycles.
        This matters for degradation accounting.
        """
        total_energy_charged = np.sum(charge_mw) * self.dt  # MWh
        return total_energy_charged / self.battery.usable_capacity_mwh

    def _annualize_revenue(self, total_revenue: float, n_periods: int) -> float:
        """
        Annualize revenue to $/kW-yr for comparability.

        $/kW-yr is the standard metric for comparing storage assets across
        different power ratings. Divide total $ by (power_mw * 1000) to
        get $/kW, then scale to a full year.
        """
        hours_in_year = 8760.0
        fraction_of_year = (n_periods * self.dt) / hours_in_year
        if fraction_of_year == 0:
            return 0.0
        annual_revenue = total_revenue / fraction_of_year
        return annual_revenue / (self.battery.power_mw * 1000)  # $/kW-yr
