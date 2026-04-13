"""
Battery Energy Storage System (BESS) model.

This module defines the physical parameters and constraints of the battery.
The BatteryConfig dataclass is the single source of truth for all physical
parameters — every optimizer reads from this, ensuring consistency.

Key design decision: separating the *physics* (this file) from the
*optimization logic* (optimizer files). This mirrors how real systems work:
the physical asset doesn't change, but dispatch strategies do.
"""

from dataclasses import dataclass, field


@dataclass
class BatteryConfig:
    """
    Physical parameters for a Battery Energy Storage System.

    All parameters use SI-adjacent units standard in energy markets:
    - Power in MW
    - Energy in MWh
    - Efficiency as a fraction (0-1)
    - Costs in $/MWh

    Reference battery (from learning pathway):
        100 MW / 400 MWh, η_c=0.93, η_d=0.95, degradation=$10/MWh,
        SOC limits 5%-95%.

    The "4-hour battery" (400 MWh / 100 MW) is the current US market
    standard for grid-scale storage. Duration matters because it determines
    how many full price spread cycles the battery can capture per day.
    """

    # --- Power limits ---
    power_mw: float = 100.0          # Maximum charge or discharge rate (MW)

    # --- Energy capacity ---
    capacity_mwh: float = 400.0      # Total nameplate energy capacity (MWh)

    # --- State of Charge (SOC) limits ---
    soc_min_pct: float = 0.05        # Minimum SOC as fraction of capacity
    soc_max_pct: float = 0.95        # Maximum SOC as fraction of capacity

    # --- Round-trip efficiency ---
    rte_charge: float = 0.93         # Charging efficiency (fraction)
    rte_discharge: float = 0.95      # Discharging efficiency (fraction)

    # --- Degradation cost ---
    degradation_cost: float = 10.0   # $/MWh of throughput (charge + discharge)

    # --- Initial conditions ---
    soc_initial_pct: float = 0.50    # Starting SOC (fraction of capacity)

    # --- Derived properties (computed from above) ---
    @property
    def soc_min_mwh(self) -> float:
        """Minimum SOC in MWh."""
        return self.soc_min_pct * self.capacity_mwh

    @property
    def soc_max_mwh(self) -> float:
        """Maximum SOC in MWh."""
        return self.soc_max_pct * self.capacity_mwh

    @property
    def soc_initial_mwh(self) -> float:
        """Initial SOC in MWh."""
        return self.soc_initial_pct * self.capacity_mwh

    @property
    def usable_capacity_mwh(self) -> float:
        """Usable energy capacity (between min and max SOC)."""
        return self.soc_max_mwh - self.soc_min_mwh

    @property
    def roundtrip_efficiency(self) -> float:
        """
        Combined round-trip efficiency.

        If you charge 1 MWh in, you get rte_charge MWh stored.
        When you discharge, you get rte_discharge of what's stored out.
        Round-trip: rte_charge * rte_discharge.

        Example: 0.93 * 0.95 = 0.8835 → ~12% losses per cycle.
        This is why you need a price spread of at least degradation_cost /
        roundtrip_efficiency to make money.
        """
        return self.rte_charge * self.rte_discharge

    def minimum_profitable_spread(self) -> float:
        """
        The minimum price spread ($/MWh) needed to profit from one cycle,
        accounting for round-trip efficiency losses and degradation cost.

        Derivation:
        - Buy 1 MWh at price p_low → pay p_low
        - Store: get rte_charge MWh in battery
        - Discharge: get rte_charge * rte_discharge MWh out = RTE MWh
        - Sell at p_high → receive p_high * RTE
        - Pay degradation: degradation_cost * (1 + RTE) for charge + discharge throughput

        Profit per MWh charged = p_high * RTE - p_low - degradation_cost * (1 + RTE)
        Set profit = 0 and solve for (p_high - p_low):
            p_high - p_low = p_low * (1 - RTE) / RTE + degradation_cost * (1 + RTE) / RTE

        Simplified minimum spread (ignoring the (1-RTE)/RTE term on p_low):
            ≈ degradation_cost * (1 + 1/RTE)

        This is a useful quick sanity check: if the daily price spread is
        below this threshold, the battery should do nothing.
        """
        rte = self.roundtrip_efficiency
        return self.degradation_cost * (1 + 1 / rte)

    def __repr__(self) -> str:
        return (
            f"BatteryConfig("
            f"{self.power_mw}MW / {self.capacity_mwh}MWh, "
            f"RTE={self.roundtrip_efficiency:.1%}, "
            f"degradation=${self.degradation_cost}/MWh)"
        )
