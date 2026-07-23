"""NIST AM-Bench 2022 single-track melt-pool dataset for IN718.

Source: Weaver, J.S., Deisenroth, D., Mekhontsev, S., Lane, B.M., Levine, L.E.,
and Yeung, H. (2024), "Cross-Sectional Melt Pool Geometry of Laser Scanned
Tracks and Pads on Nickel Alloy 718 for the 2022 Additive Manufacturing
Benchmark Challenges," Integrating Materials and Manufacturing Innovation,
13(2), doi: 10.1007/s40192-024-00355-5. Table 4 (single-track cases).

Underlying measurement dataset: NIST AM Bench 2022, doi: 10.18434/mds2-2718.

This is the NIST AM-Bench benchmark series -- the community-standard,
metrology-grade reference measurements used across the L-PBF process-modelling
literature specifically because they are independently produced, rigorously
characterized (including stated measurement uncertainty), and not affiliated
with any single research group's own modelling methodology.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NistAmBenchCase:
    """One single-track case from Weaver et al. (2024), Table 4."""

    case_id: str
    power: float  # W
    velocity: float  # m/s
    spot_diameter: float  # m
    measured_width: float  # m
    measured_depth: float  # m


NIST_AMBENCH_2022_IN718: list[NistAmBenchCase] = [
    NistAmBenchCase("0", 285.0, 0.960, 67e-6, 136.3e-6, 139.7e-6),
    NistAmBenchCase("1.1", 285.0, 0.960, 49e-6, 106.2e-6, 227.2e-6),
    NistAmBenchCase("1.2", 285.0, 0.960, 82e-6, 141.7e-6, 102.4e-6),
    NistAmBenchCase("2.1", 285.0, 1.200, 67e-6, 112.9e-6, 109.7e-6),
    NistAmBenchCase("2.2", 285.0, 0.800, 67e-6, 156.1e-6, 176.5e-6),
    NistAmBenchCase("3.1", 325.0, 0.960, 67e-6, 134.3e-6, 166.1e-6),
    NistAmBenchCase("3.2", 245.0, 0.960, 67e-6, 129.4e-6, 116.9e-6),
]
