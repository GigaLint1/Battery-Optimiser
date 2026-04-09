"""
Tests for the LP optimizer.

These tests encode the validation exercises from the learning pathway.
They serve two purposes:
  1. Verify your LP implementation is correct
  2. Build intuition — each test checks a claim about how the battery *should*
     behave under specific market conditions. Before running the test, predict
     the outcome. If the test fails, ask why before debugging.

Run with: pytest tests/test_lp_optimizer.py -v
"""

import numpy as np
import pytest
from battery.battery import BatteryConfig
from battery.lp_optimizer import LPOptimizer, LPResult


@pytest.fixture
def default_battery() -> BatteryConfig:
    """Standard 100MW/400MWh battery from the learning pathway."""
    return BatteryConfig(
        power_mw=100.0,
        capacity_mwh=400.0,
        eta_charge=0.93,
        eta_discharge=0.95,
        degradation_cost=10.0,
        soc_min_pct=0.05,
        soc_max_pct=0.95,
        soc_initial_pct=0.50,
    )


@pytest.fixture
def optimizer(default_battery) -> LPOptimizer:
    return LPOptimizer(default_battery)


class TestBatteryConfig:
    """Basic sanity checks on the BatteryConfig dataclass."""

    def test_derived_properties(self, default_battery):
        bat = default_battery
        assert bat.soc_min_mwh == pytest.approx(20.0)   # 5% of 400
        assert bat.soc_max_mwh == pytest.approx(380.0)  # 95% of 400
        assert bat.usable_capacity_mwh == pytest.approx(360.0)
        assert bat.roundtrip_efficiency == pytest.approx(0.93 * 0.95)

    def test_minimum_profitable_spread(self, default_battery):
        """
        The minimum spread should be strictly positive.
        Any spread below this means the battery shouldn't cycle at all.
        """
        min_spread = default_battery.minimum_profitable_spread()
        assert min_spread > 0
        # With degradation=$10 and RTE=0.8835, roughly $21-23/MWh
        assert 15 < min_spread < 30


class TestLPOptimizerMonotonicPrices:
    """
    Validation 1: Monotonically increasing prices.

    If prices strictly increase throughout the day, the optimal strategy
    is trivial: charge fully at the start, discharge fully at the end.

    PREDICTION: Before running this test, predict:
    - When does the battery start discharging?
    - What is the final SOC?
    - Is there any period where both charge and discharge are non-zero?
    """

    def test_charges_at_start_discharges_at_end(self, optimizer, default_battery):
        # Prices strictly increase from $10 to $100 over 24 hours
        prices = np.linspace(10, 100, 24)
        result = optimizer.solve(prices)

        assert result.status == "optimal", f"LP failed: {result.status}"

        # The battery should charge in the first half and discharge in the second
        # (exact cutpoint depends on degradation cost and efficiency)
        # At minimum: net discharge > 0 in the latter hours
        net_power = result.discharge_mw - result.charge_mw
        assert np.any(net_power > 0.1), "Battery never discharges"
        assert np.any(net_power < -0.1), "Battery never charges"

    def test_never_simultaneous_charge_discharge(self, optimizer):
        """
        The LP should never simultaneously charge and discharge (for positive prices).
        This is not explicitly constrained — it falls out of the objective naturally.

        THINK: Why does the objective automatically prevent this?
        """
        prices = np.linspace(10, 100, 24)
        result = optimizer.solve(prices)

        # For each period, at least one of charge or discharge should be ~zero
        for t in range(len(prices)):
            c = result.charge_mw[t]
            d = result.discharge_mw[t]
            assert min(c, d) < 1e-4, (
                f"Simultaneous charge ({c:.2f}MW) and discharge ({d:.2f}MW) at t={t}"
            )


class TestLPOptimizerFlatPrices:
    """
    Validation 2: Flat prices.

    If prices are constant, no arbitrage is possible. The battery should
    do nothing (because any cycling incurs degradation cost with zero revenue gain).

    PREDICTION: What revenue do you expect? What is the optimal schedule?
    """

    def test_does_nothing_with_flat_prices(self, optimizer):
        prices = np.full(24, 50.0)  # $50/MWh all day
        result = optimizer.solve(prices)

        assert result.status == "optimal"
        # Revenue should be zero (or slightly negative if numerical noise)
        assert result.total_revenue == pytest.approx(0.0, abs=1.0)
        # Charge and discharge should both be near zero
        assert np.all(result.charge_mw < 1e-3)
        assert np.all(result.discharge_mw < 1e-3)


class TestLPOptimizerNegativePrices:
    """
    Validation 3: Negative price periods.

    During hours of excess solar generation, prices can go negative.
    A battery should CHARGE during negative price hours (you get paid to consume).
    This tests the pathological case that breaks the "no simultaneous
    charge/discharge" intuition.

    THINK: With a price of -$50/MWh:
    - You get PAID $50 for every MWh you consume (charge)
    - Does the LP correctly identify this as a revenue opportunity?
    """

    def test_charges_during_negative_prices(self, optimizer):
        # Prices: negative midday, high in evening
        prices = np.array([
            30, 30, 30, 30, 30, 30,   # overnight
            30, 30, 30, 30, 30, -20,  # morning into solar
            -50, -50, -20, 10, 30, 80, # midday glut
            120, 100, 70, 50, 35, 30,  # evening peak
        ], dtype=float)

        result = optimizer.solve(prices)
        assert result.status == "optimal"

        # Battery should charge during negative price hours (11-13)
        assert result.charge_mw[12] > 1.0 or result.charge_mw[11] > 1.0, (
            "Battery should charge during negative price hours"
        )

    def test_simultaneous_charge_discharge_with_negative_prices(self, optimizer):
        """
        ADVANCED: When prices are negative, can both c_t and d_t be positive?

        Think carefully: if price is -$50/MWh, you get paid to charge AND
        you could simultaneously discharge (sell) at -$50 (you pay to sell?).
        The LP might behave unexpectedly here.

        This is the known failure mode. What does it reveal about the LP formulation?
        """
        # All prices are negative — degenerate case
        prices = np.full(24, -10.0)
        result = optimizer.solve(prices)

        # Document what happens here — this is a learning exercise
        # The correct behavior: charge as much as possible (get paid to consume)
        # But does the LP also try to discharge simultaneously?
        print(f"\nWith all-negative prices:")
        print(f"  Total charge: {result.charge_mw.sum():.1f} MWh")
        print(f"  Total discharge: {result.discharge_mw.sum():.1f} MWh")
        print(f"  Total revenue: ${result.total_revenue:.2f}")
        # No assertion — this is for you to analyze


class TestLPSOCConstraints:
    """Tests that SOC constraints are respected."""

    def test_soc_within_bounds(self, optimizer, default_battery):
        prices = np.linspace(10, 100, 24)
        result = optimizer.solve(prices)

        bat = default_battery
        assert np.all(result.soc_mwh >= bat.soc_min_mwh - 1e-4)
        assert np.all(result.soc_mwh <= bat.soc_max_mwh + 1e-4)

    def test_soc_dynamics_satisfied(self, optimizer, default_battery):
        """
        Verify the SOC transition equation holds exactly.

        SOC_{t+1} = SOC_t + η_c * charge_t * dt - (1/η_d) * discharge_t * dt

        The (1/η_d) on discharge is crucial: to deliver 1 MWh to the grid,
        the battery must release 1/η_d MWh from storage. This asymmetry
        is often implemented incorrectly.
        """
        prices = np.linspace(10, 100, 24)
        result = optimizer.solve(prices)

        bat = default_battery
        dt = optimizer.dt

        for t in range(len(prices)):
            expected_soc = (
                result.soc_mwh[t]
                + bat.eta_charge * result.charge_mw[t] * dt
                - (1 / bat.eta_discharge) * result.discharge_mw[t] * dt
            )
            actual_soc = result.soc_mwh[t + 1]
            assert expected_soc == pytest.approx(actual_soc, abs=1e-4), (
                f"SOC dynamics violated at t={t}: "
                f"expected {expected_soc:.4f}, got {actual_soc:.4f}"
            )


class TestLPDualVariables:
    """
    Tests for dual variable extraction.

    Dual variables are the economic heart of LP optimization.
    The dual on the SOC constraint at time t = marginal value of an
    additional MWh of storage capacity at that moment.
    """

    def test_dual_variables_extracted(self, optimizer):
        prices = np.linspace(10, 100, 24)
        result = optimizer.solve(prices)

        assert result.dual_soc is not None, (
            "Dual variables on SOC constraint should be extracted"
        )
        assert len(result.dual_soc) == len(prices), (
            "Should have one dual variable per time period"
        )

    def test_dual_interpretation(self, optimizer):
        """
        The dual variable on SOC at hour t should be highest during the
        peak price periods — that's when storage capacity is most valuable.

        THINK: Why is this the economic interpretation?
        """
        prices = np.linspace(10, 100, 24)
        result = optimizer.solve(prices)

        if result.dual_soc is not None:
            # Peak value of storage should be in the higher-price hours
            peak_dual_hour = np.argmax(np.abs(result.dual_soc))
            assert peak_dual_hour > 12, (
                f"Dual value peaks at hour {peak_dual_hour}, expected late-day "
                f"when prices are highest"
            )
