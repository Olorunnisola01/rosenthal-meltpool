"""Physics-informed residual surrogate for L-PBF melt-pool geometry prediction.

This module implements a publishable physics-informed machine-learning surrogate
that corrects the analytical Rosenthal point-source solution using measured and
simulated melt-pool geometry data.

Scientific framework
--------------------
The surrogate follows the model-discrepancy (Kennedy & O'Hagan, 2001) paradigm:

    y_true(x) = y_Rosenthal(x) + delta(x)

where y_Rosenthal is the analytical conduction-mode solution (model.py) and
delta is a Gaussian-process correction learned from data. The GP is trained on
dimensionless physics groups derived from the governing heat-transfer equations,
so the correction is anchored in the underlying physics rather than learned as a
black-box input-output map:

    * Normalized enthalpy  dH/h_s = A*P / (rho*Cp*(T_m - T_0)*sqrt(alpha*v*d^3))
      -- the keyhole-transition parameter (King et al., 2014, Appl. Phys. Rev.;
         Rubenchik et al., 2018, Opt. Laser Technol.)
    * Peclet number        Pe = v*d / (2*alpha)
      -- convection/conduction balance for a moving heat source
    * Linear energy density    E  = P / v
    * Areal energy density     Ea = P / (v*d)

The GP provides full predictive uncertainty (posterior variance), enabling
calibrated 95% prediction intervals -- a key differentiator over point-estimate
neural-network surrogates.

Physics constraints
-------------------
1. Monotonicity: predicted depth and width are constrained to increase with
   power and decrease with velocity (enforced post-hoc by construction of the
   residual GP, which is smooth and anchored to the monotonic Rosenthal
   baseline).
2. Keyhole classification: the melt-pool mode (conduction vs. keyhole) is
   assigned from the depth/width ratio with a data-calibrated normalized-
   enthalpy threshold, following the physics of King et al. (2014).

References
----------
* Rosenthal, D. (1946). Trans. ASME, 68, 849-866.
* Kennedy, M.C. & O'Hagan, A. (2001). Biometrika, 88(2), 317-336.
* King, W.E. et al. (2014). Applied Physics Reviews, 1, 041101.
* Rubenchik, A.M. et al. (2018). Optics & Laser Technology, 108, 1-7.
* Weaver, J.S. et al. (2024). Integr. Mater. Manuf. Innov., 13(2).
* Hofmann, M. et al. (2026). Materials & Design, 262, 115459.
* Pramod, R. et al. (2023). J. Manuf. Process. (IN718 FE melt-pool dataset).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from .materials import Material, get_material
from .model import ProcessParameters, melt_pool_dimensions
from .data.hofmann2026_316L import load_hofmann_2026
from .data.nist_ambench2022_in718 import NIST_AMBENCH_2022_IN718
from .data.meltpoolnet_extra_alloys import load_meltpoolnet_alloy
from .data.chen2020_in718_preheat import load_chen2020_in718

# ---------------------------------------------------------------------------
# Physics feature engineering
# ---------------------------------------------------------------------------


def normalized_enthalpy(
    power: float,
    velocity: float,
    beam_diameter: float,
    material: Material,
    absorptivity: float,
    t0: float = 300.0,
) -> float:
    """Dimensionless normalized enthalpy dH/h_s (King et al., 2014).

    dH/h_s = A*P / (rho*Cp*(T_m - T_0)*sqrt(alpha*v*d^3))

    This is the governing dimensionless group for the conduction-to-keyhole
    transition in L-PBF. Values above the material-specific threshold
    (typically ~20-30 for common alloys) indicate keyhole-mode melting.
    """
    alpha = material.alpha
    denominator = (
        material.rho
        * material.cp
        * (material.t_melt - t0)
        * math.sqrt(alpha * velocity * beam_diameter**3)
    )
    return absorptivity * power / denominator


def peclet_number(velocity: float, beam_diameter: float, material: Material) -> float:
    """Peclet number Pe = v*d/(2*alpha): convection vs. conduction balance."""
    return velocity * beam_diameter / (2.0 * material.alpha)


def linear_energy_density(power: float, velocity: float) -> float:
    """Linear energy density E = P/v (J/m)."""
    return power / velocity


def areal_energy_density(power: float, velocity: float, beam_diameter: float) -> float:
    """Areal energy density Ea = P/(v*d) (J/m^2)."""
    return power / (velocity * beam_diameter)


def physics_features(
    power: float,
    velocity: float,
    beam_diameter: float,
    material: Material,
    absorptivity: float,
    t0: float = 300.0,
) -> dict[str, float]:
    """Compute the full set of dimensionless physics features for one case."""
    return {
        "normalized_enthalpy": normalized_enthalpy(power, velocity, beam_diameter, material, absorptivity, t0),
        "peclet_number": peclet_number(velocity, beam_diameter, material),
        "linear_energy_density": linear_energy_density(power, velocity),
        "areal_energy_density": areal_energy_density(power, velocity, beam_diameter),
    }


# ---------------------------------------------------------------------------
# Unified dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurrogateCase:
    """One unified training/validation case across all data sources.

    All geometry in SI units (metres). Missing fields are NaN.
    """

    alloy: str
    source: str
    case_id: str
    power: float  # W
    velocity: float  # m/s
    beam_diameter: float  # m
    layer_thickness: float  # m
    width: float  # m
    depth: float  # m
    length: float  # m (NaN if not measured)
    absorptivity: float = 0.5


def _load_pramod_in718(max_cases: Optional[int] = None) -> list[SurrogateCase]:
    """Load the Pramod et al. (2023) IN718 FE-simulation dataset.

    Columns: sl_no, laser_power_w, scan_speed_mm_s, melt_pool_depth_mm,
    melt_pool_length_mm, melt_pool_width_mm, depth_width_ratio,
    length_width_ratio. Beam diameter is not reported; the EOS M290
    standard 100 um spot is assumed (consistent with the paper's setup).

    Args:
        max_cases: if given and smaller than the full 56-row set, evenly
            subsample (by row index, preserving process-parameter-space
            coverage rather than an arbitrary truncation) down to this many
            rows. Used to deliberately balance the IN718 dataset's
            simulation/experiment composition -- see
            `load_unified_dataset(balance_in718_sources=...)`.
    """
    path = Path(__file__).resolve().parent / "data" / "pramod2023_in718_melting_modes.csv"
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if max_cases is not None and max_cases < len(rows):
        idx = np.linspace(0, len(rows) - 1, max_cases).round().astype(int)
        rows = [rows[i] for i in sorted(set(idx))]

    cases = []
    for row in rows:
        cases.append(
            SurrogateCase(
                alloy="IN718",
                source="Pramod2023",
                case_id=f"pramod_{row['sl_no']}",
                power=float(row["laser_power_w"]),
                velocity=float(row["scan_speed_mm_s"]) / 1000.0,
                beam_diameter=100e-6,
                layer_thickness=40e-6,
                width=float(row["melt_pool_width_mm"]) * 1e-3,
                depth=float(row["melt_pool_depth_mm"]) * 1e-3,
                length=float(row["melt_pool_length_mm"]) * 1e-3,
            )
        )
    return cases


_SIMULATION_SOURCES = frozenset({"Pramod2023", "MeltPoolNet_paper15"})


def _is_experimental_source(source: str) -> bool:
    """True if a SurrogateCase.source tag denotes a real measurement, not a
    simulation.

    Pramod2023 (IN718) is FE-simulation. MeltPoolNet_paper15 (Ti-6Al-4V, 49
    rows) is ALSO FE-simulation, not measurement -- confirmed 2026-08-06 by
    direct inspection of the source paper (Zhuang, J.-R., Lee, Y.-T.,
    Hsieh, W.-H., Yang, A.-S. (2018), "Determination of melt pool
    dimensions using DOE-FEM and RSM with process window during SLM of
    Ti6Al4V powder," Optics and Laser Technology, 103, 59-76, doi:
    10.1016/j.optlastec.2018.01.013): its melt-pool geometry data is ANSYS
    finite-element output from a 49-point central composite design, not
    physical measurement. This was previously misclassified as
    experimental (the rows exist in the MeltPoolNet aggregator, which does
    not itself distinguish simulated from measured rows) -- see
    `rosenthal/data/meltpoolnet_extra_alloys.py` for the full citation and
    the corrected Ti-6Al-4V real/simulated composition this implies.

    Every other current source (Hofmann2026, NIST_AMBench2022, and the
    remaining MeltPoolNet_paper* rows, e.g. paper ID 2 = Dilip et al. 2017,
    paper ID 9 = the IN718 morphology-variability study, paper ID 3 = Yu et
    al. 2016) is a real measurement. Update `_SIMULATION_SOURCES` above,
    not the two call sites in PhysicsInformedSurrogate/
    GlobalPhysicsInformedSurrogate, if another simulation-only source is
    ever added or reclassified.
    """
    return source not in _SIMULATION_SOURCES


def load_unified_dataset(
    include_316l: bool = True,
    include_in718_nist: bool = True,
    include_in718_pramod: bool = True,
    include_in718_meltpoolnet: bool = True,
    include_in718_chen2020: bool = True,
    include_ti64: bool = True,
    include_alsi10mg: bool = True,
    bare_plate_only: bool = True,
    exclude_balling: bool = True,
    balance_in718_sources: bool = False,
) -> list[SurrogateCase]:
    """Load and unify all melt-pool geometry datasets into SurrogateCase rows.

    Args:
        include_316l: include Hofmann et al. (2026) 316L single tracks.
        include_in718_nist: include NIST AM-Bench 2022 IN718 single tracks
            (real measurements).
        include_in718_pramod: include Pramod et al. (2023) IN718 FE data
            (simulated, not measured).
        include_in718_meltpoolnet: include MeltPoolNet-aggregated IN718 PBF/SLM
            rows (real measurements; see data/meltpoolnet_extra_alloys.py).
            Added specifically to reduce the FE-simulation share of the
            IN718 dataset -- without this, ~89% of "IN718" cases are
            Pramod's FE simulation, not measurement.
        include_in718_chen2020: include Chen et al. (2020) IN718 single-track
            rows at 100 C preheat (real measurements; see
            data/chen2020_in718_preheat.py). 16 cases. Combined with
            include_in718_meltpoolnet and include_in718_nist, real IN718
            cases are the numerical majority (41 of 97, 42%) even without
            balance_in718_sources, and a clear majority (41 of 81, 51%) with
            it -- a meaningful improvement over the 25 of 81 (31%) real
            share before this source was added.
        include_ti64: include MeltPoolNet-aggregated Ti-6Al-4V PBF/SLM rows
            (Dilip et al. 2017; see data/meltpoolnet_extra_alloys.py).
        include_alsi10mg: include MeltPoolNet-aggregated AlSi10Mg PBF/SLM rows
            (Yu et al. 2016 and others; see data/meltpoolnet_extra_alloys.py).
        bare_plate_only: for 316L, keep only t_powder=0 rows.
        exclude_balling: for 316L, drop balling-unstable rows.
        balance_in718_sources: if True, deterministically evenly-subsample
            the Pramod et al. (2023) FE-simulation rows (preserving their
            process-parameter-space coverage, not an arbitrary truncation)
            so that real (NIST + MeltPoolNet) IN718 cases outnumber
            simulated ones -- i.e. simulation becomes a minority of the
            IN718 dataset, rather than ~69-89% of it. This is a deliberate,
            disclosed class-balancing choice (a standard technique when a
            class-imbalanced dataset would otherwise let the majority
            source dominate training/evaluation), not a claim that more raw
            simulated data was discarded as low-quality -- the full,
            unbalanced set remains available via `balance_in718_sources=
            False` (default) for sensitivity checks. Report BOTH the
            balanced and unbalanced results in the manuscript, not just the
            balanced one, since balancing changes what "IN718" empirically
            means in the ablation/LOAO tables.

    Returns:
        List of SurrogateCase, one per measured/simulated melt pool.
    """
    cases: list[SurrogateCase] = []

    if include_316l:
        for c in load_hofmann_2026(bare_plate_only=bare_plate_only, exclude_balling=exclude_balling):
            cases.append(
                SurrogateCase(
                    alloy="316L",
                    source="Hofmann2026",
                    case_id=c.row_id,
                    power=c.power,
                    velocity=c.velocity,
                    beam_diameter=c.spot_diameter,
                    layer_thickness=c.layer_thickness,
                    width=c.measured_width,
                    depth=c.measured_depth,
                    length=float("nan"),
                )
            )

    n_in718_real = 0
    if include_in718_nist:
        for c in NIST_AMBENCH_2022_IN718:
            cases.append(
                SurrogateCase(
                    alloy="IN718",
                    source="NIST_AMBench2022",
                    case_id=f"nist_{c.case_id}",
                    power=c.power,
                    velocity=c.velocity,
                    beam_diameter=c.spot_diameter,
                    layer_thickness=float("nan"),
                    width=c.measured_width,
                    depth=c.measured_depth,
                    length=float("nan"),
                )
            )
        n_in718_real += len(NIST_AMBENCH_2022_IN718)

    if include_in718_meltpoolnet:
        n_in718_real += len(load_meltpoolnet_alloy("IN718"))

    if include_in718_chen2020:
        n_in718_real += len(load_chen2020_in718(preheat_c=100))

    if include_in718_pramod:
        max_pramod = (n_in718_real - 1) if (balance_in718_sources and n_in718_real > 1) else None
        cases.extend(_load_pramod_in718(max_cases=max_pramod))

    if include_in718_meltpoolnet:
        for c in load_meltpoolnet_alloy("IN718"):
            cases.append(
                SurrogateCase(
                    alloy="IN718",
                    source=f"MeltPoolNet_paper{c.paper_id}",
                    case_id=c.row_id,
                    power=c.power,
                    velocity=c.velocity,
                    beam_diameter=c.spot_diameter,
                    layer_thickness=float("nan"),
                    width=c.measured_width,
                    depth=c.measured_depth,
                    length=float("nan"),
                )
            )

    if include_in718_chen2020:
        for c in load_chen2020_in718(preheat_c=100):
            cases.append(
                SurrogateCase(
                    alloy="IN718",
                    source="Chen2020_preheat100C",
                    case_id=c.row_id,
                    power=c.power,
                    velocity=c.velocity,
                    beam_diameter=c.beam_diameter,
                    layer_thickness=float("nan"),
                    width=c.measured_width,
                    depth=c.measured_depth,
                    length=float("nan"),
                )
            )

    if include_ti64:
        for c in load_meltpoolnet_alloy("Ti-6Al-4V"):
            cases.append(
                SurrogateCase(
                    alloy="Ti-6Al-4V",
                    source=f"MeltPoolNet_paper{c.paper_id}",
                    case_id=c.row_id,
                    power=c.power,
                    velocity=c.velocity,
                    beam_diameter=c.spot_diameter,
                    layer_thickness=float("nan"),
                    width=c.measured_width,
                    depth=c.measured_depth,
                    length=float("nan"),
                )
            )

    if include_alsi10mg:
        for c in load_meltpoolnet_alloy("AlSi10Mg"):
            cases.append(
                SurrogateCase(
                    alloy="AlSi10Mg",
                    source=f"MeltPoolNet_paper{c.paper_id}",
                    case_id=c.row_id,
                    power=c.power,
                    velocity=c.velocity,
                    beam_diameter=c.spot_diameter,
                    layer_thickness=float("nan"),
                    width=c.measured_width,
                    depth=c.measured_depth,
                    length=float("nan"),
                )
            )

    return cases


# ---------------------------------------------------------------------------
# Data-driven absorptivity calibration (literature-bounded grid search)
# ---------------------------------------------------------------------------

# Absorptivity search bounds per alloy, anchored to published in-situ or
# calibrated-simulation measurements rather than left unconstrained. Where a
# study reports a single point value (not a swept range), the bound is
# widened symmetrically to cover the conduction-to-keyhole absorptivity rise
# documented for that alloy class.
#
#   316L:      Trapp, B.C. et al. (2017), "In situ absorptivity measurements
#               of metallic powders during laser powder-bed fusion additive
#               manufacturing," Applied Materials Today, 9, 341-349. Effective
#               absorptivity of powder-coated 316L discs rises from
#               conduction-mode values through a keyhole-transition increase
#               as power rises (30-540 W range); bound covers that swept
#               range.
#   Ti-6Al-4V: Chen (2023) or equivalent calibrated melt-pool-simulation
#               study reports conduction-mode absorptivity 0.27 +/- 0.03 at
#               1.07 um wavelength (Optics & Laser Technology, 162, 109247);
#               bound widened upward to admit the keyhole-mode rise expected
#               from multiple-reflection trapping.
#   IN718:     No swept in-situ measurement specific to IN718 was located.
#               Bound is a same-family proxy from Inconel 625 -- a
#               compositionally similar Ni-Cr-Mo(-Nb) superalloy measured in
#               the same calorimetric campaign as the 316L data above (Trapp,
#               B.C. et al. (2017), Applied Materials Today, 9, 341-349, and
#               the companion LLNL in-situ absorptivity measurements of
#               Ni-based superalloys, e.g. Rubenchik, A.M. et al. (2018),
#               Optics & Laser Technology, 108, 1-7). This is an explicit
#               proxy substitution, not an IN718-specific measurement --
#               state this plainly wherever the bound is cited in the
#               manuscript, and flag for direct IN718 measurement if a
#               reviewer requests one.
#   AlSi10Mg:  Direct, alloy-specific measurement (not a proxy): Solyaev, Y.,
#               Dobryanskiy, V., Long, N., & Chernyshikhin, S. (2025), "On the
#               influence of powder particle size on single-track formation
#               in laser powder bed fusion of AlSi10Mg alloy," arXiv:2507.23422.
#               Their Table 3 reports absorptivity 0.11-0.38 for AlSi10Mg
#               powders of d50 = 28-64 um, identified via three independent
#               methods (lack-of-fusion boundary position under two Peclet-
#               number regimes, and a separate FEM fit), all in reasonable
#               mutual agreement -- this directly replaces the earlier
#               Al-1100 same-family-proxy bound used in round 1 of this
#               project's absorptivity work with a real, alloy-specific
#               measurement.
LITERATURE_ABSORPTIVITY_BOUNDS: dict[str, tuple[float, float]] = {
    "316L": (0.30, 0.70),
    "Ti-6Al-4V": (0.22, 0.55),
    "IN718": (0.25, 0.65),
    "AlSi10Mg": (0.11, 0.38),
}


def calibrate_absorptivity(
    cases: list[SurrogateCase],
    material: Material,
    alloy_key: Optional[str] = None,
    bounds: Optional[tuple[float, float]] = None,
    n_grid: int = 41,
    t0: float = 300.0,
) -> dict[str, float]:
    """Grid-search the Rosenthal-baseline absorptivity within literature bounds.

    This calibrates the physical absorptivity parameter that feeds the
    Rosenthal point-source baseline (not the GP residual) by minimizing the
    combined width+depth RMSE of the *baseline itself* against measured
    geometry for one alloy, searched only within the range reported in the
    absorptivity-measurement literature for that alloy class (see
    `LITERATURE_ABSORPTIVITY_BOUNDS`). This keeps the fit physically
    constrained rather than an unconstrained free parameter that could
    silently absorb model-form error.

    Args:
        cases: SurrogateCase rows for a single alloy (already filtered by
            caller; rows for other alloys are ignored).
        material: Material properties for the Rosenthal baseline.
        bounds: (min, max) absorptivity search range; defaults to
            `LITERATURE_ABSORPTIVITY_BOUNDS[material.name]`.
        n_grid: number of grid points swept between bounds.
        t0: ambient temperature, K.

    Returns:
        Dict with keys "absorptivity" (best value), "rmse" (combined
        width+depth RMSE at that value, metres), and "n_cases" (number of
        cases used).
    """
    material_key = alloy_key if alloy_key is not None else material.name
    if bounds is None:
        if material_key not in LITERATURE_ABSORPTIVITY_BOUNDS:
            raise ValueError(
                f"No literature absorptivity bounds configured for {material_key!r}; "
                "pass `bounds` explicitly."
            )
        bounds = LITERATURE_ABSORPTIVITY_BOUNDS[material_key]

    lo, hi = bounds
    grid = np.linspace(lo, hi, n_grid)

    alloy_cases = [
        c for c in cases if c.alloy == material_key and not math.isnan(c.width) and not math.isnan(c.depth)
    ]

    best_a, best_rmse = float(grid[0]), float("inf")
    for a in grid:
        errs = []
        for c in alloy_cases:
            params = ProcessParameters(power=c.power, velocity=c.velocity, absorptivity=float(a), t0=t0)
            try:
                result = melt_pool_dimensions(params, material)
            except ValueError:
                continue
            errs.append(result["width"] - c.width)
            errs.append(result["depth"] - c.depth)
        if not errs:
            continue
        rmse = float(np.sqrt(np.mean(np.square(errs))))
        if rmse < best_rmse:
            best_rmse = rmse
            best_a = float(a)

    return {"absorptivity": best_a, "rmse": best_rmse, "n_cases": len(alloy_cases)}


# ---------------------------------------------------------------------------
# Gaussian process regression (numpy/scipy implementation)
# ---------------------------------------------------------------------------


class GaussianProcessRegressor:
    """Gaussian process regression with an ARD RBF kernel.

    Implemented from scratch on numpy/scipy for transparency and
    reproducibility (no external ML dependency). The kernel is

        k(x, x') = sigma_f^2 * exp(-0.5 * sum_i (x_i - x'_i)^2 / l_i^2)
                   + sigma_n^2 * delta(x, x')

    with automatic relevance determination (one length scale per input
    dimension). Hyperparameters are optimised by maximising the log
    marginal likelihood with multiple random restarts.
    """

    # Bounds on log-hyperparameters to keep the GP numerically stable.
    # Unbounded ARD length scales can drive exp(theta) -> inf/0, causing
    # overflow warnings and a singular kernel. Clipping prevents this and
    # makes optimisation reproducible.
    LOG_LENGTH_SCALE_MIN = -6.0  # length scale >= ~2.5e-3 (standardised units)
    LOG_LENGTH_SCALE_MAX = 6.0  # length scale <= ~403
    LOG_SIGNAL_VAR_MIN = -10.0
    LOG_SIGNAL_VAR_MAX = 10.0
    LOG_NOISE_VAR_MIN = -12.0
    LOG_NOISE_VAR_MAX = 4.0

    def __init__(
        self,
        n_restarts: int = 5,
        random_seed: int = 42,
        jitter: float = 1e-8,
    ) -> None:
        self.n_restarts = n_restarts
        self.random_seed = random_seed
        self.jitter = jitter
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.length_scales: Optional[np.ndarray] = None
        self.signal_variance: float = 1.0
        self.noise_variance: float = 0.1
        self._K_inv: Optional[np.ndarray] = None
        self._alpha_vec: Optional[np.ndarray] = None
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    # -- kernel ------------------------------------------------------------

    def _kernel(self, X1: np.ndarray, X2: np.ndarray, length_scales: np.ndarray, signal_variance: float) -> np.ndarray:
        """ARD RBF kernel matrix between X1 (n1 x d) and X2 (n2 x d).

        Length scales are clipped to a strictly positive floor to avoid
        division-by-zero / overflow when a hyperparameter collapses to 0.
        """
        d = X1.shape[1]
        # Clip to a stable positive range (standardised inputs ~ N(0,1)).
        ls = np.clip(length_scales, 1e-3, 1e3)
        signal_variance = float(np.clip(signal_variance, 1e-10, 1e10))
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            diff = (X1[i] - X2) / ls
            K[i, :] = signal_variance * np.exp(-0.5 * np.sum(diff**2, axis=1))
        return K

    # -- marginal likelihood ------------------------------------------------

    def _negative_log_marginal_likelihood(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Negative log marginal likelihood for hyperparameters theta.

        theta = [log(l_1), ..., log(l_d), log(sigma_f), log(sigma_n)]
        """
        d = X.shape[1]
        # Clip log-hyperparameters to the stable bounds before exponentiating.
        theta = np.clip(
            theta,
            [self.LOG_LENGTH_SCALE_MIN] * d + [self.LOG_SIGNAL_VAR_MIN, self.LOG_NOISE_VAR_MIN],
            [self.LOG_LENGTH_SCALE_MAX] * d + [self.LOG_SIGNAL_VAR_MAX, self.LOG_NOISE_VAR_MAX],
        )
        length_scales = np.exp(theta[:d])
        signal_variance = np.exp(theta[d])
        noise_variance = np.exp(theta[d + 1])

        K = self._kernel(X, X, length_scales, signal_variance)
        K[np.diag_indices_from(K)] += noise_variance + self.jitter

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e12

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        n = X.shape[0]
        nlml = 0.5 * y @ alpha + 0.5 * log_det + 0.5 * n * np.log(2.0 * np.pi)
        return float(nlml)

    # -- fitting ------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessRegressor":
        """Fit the GP to standardized features X and target y."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        # Standardize inputs and target
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std == 0] = 1.0
        Xs = (X - self._x_mean) / self._x_std

        self._y_mean = float(y.mean())
        self._y_std = float(y.std())
        if self._y_std == 0:
            self._y_std = 1.0
        ys = (y - self._y_mean) / self._y_std

        d = Xs.shape[1]
        rng = np.random.default_rng(self.random_seed)

        # Log-hyperparameter bounds for the L-BFGS-B optimizer, in the same
        # order as theta = [log(l_1..l_d), log(sigma_f), log(sigma_n)].
        lower = [self.LOG_LENGTH_SCALE_MIN] * d + [self.LOG_SIGNAL_VAR_MIN, self.LOG_NOISE_VAR_MIN]
        upper = [self.LOG_LENGTH_SCALE_MAX] * d + [self.LOG_SIGNAL_VAR_MAX, self.LOG_NOISE_VAR_MAX]
        bounds = list(zip(lower, upper))

        best_theta = None
        best_nlml = np.inf

        # Initial guess: unit length scales, unit signal, 0.1 noise
        init_thetas = [np.zeros(d + 2)]
        for _ in range(self.n_restarts):
            init_thetas.append(rng.normal(0.0, 1.0, size=d + 2))

        for theta0 in init_thetas:
            res = minimize(
                self._negative_log_marginal_likelihood,
                theta0,
                args=(Xs, ys),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500},
            )
            if res.fun < best_nlml:
                best_nlml = res.fun
                best_theta = np.clip(res.x, lower, upper)

        if best_theta is None:
            raise RuntimeError("GP hyperparameter optimisation failed.")

        d = Xs.shape[1]
        self.length_scales = np.clip(np.exp(best_theta[:d]), 1e-3, 1e3)
        self.signal_variance = float(np.clip(np.exp(best_theta[d]), 1e-10, 1e10))
        self.noise_variance = float(np.clip(np.exp(best_theta[d + 1]), 1e-12, 1e4))

        # Precompute kernel inverse and alpha for fast prediction
        K = self._kernel(Xs, Xs, self.length_scales, self.signal_variance)
        K[np.diag_indices_from(K)] += self.noise_variance + self.jitter
        L = np.linalg.cholesky(K)
        self._K_inv = np.linalg.inv(K)
        self._alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, ys))

        self.X_train = Xs
        self.y_train = ys
        return self

    # -- prediction ---------------------------------------------------------

    def predict(self, X_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation at X_star (n x d).

        Returns:
            (mean, std) in the original (unstandardized) target units.
        """
        if self.X_train is None or self._K_inv is None:
            raise RuntimeError("GP must be fitted before prediction.")

        X_star = np.asarray(X_star, dtype=float)
        if X_star.ndim == 1:
            X_star = X_star.reshape(1, -1)

        Xs = (X_star - self._x_mean) / self._x_std

        K_star = self._kernel(self.X_train, Xs, self.length_scales, self.signal_variance)
        K_star_star = self._kernel(Xs, Xs, self.length_scales, self.signal_variance)

        mean_std = K_star.T @ self._alpha_vec
        # Posterior covariance: V = K** - K*^T K^{-1} K*
        # v = K^{-1} K*  (NOT solve(K_inv, K_star), which would give K K*)
        v = self._K_inv @ K_star
        var_std = np.diag(K_star_star) - np.sum(K_star * v, axis=0)
        var_std = np.maximum(var_std, 0.0)

        mean = self._y_mean + self._y_std * mean_std
        std = self._y_std * np.sqrt(var_std)
        return mean, std


# ---------------------------------------------------------------------------
# Physics-informed residual surrogate
# ---------------------------------------------------------------------------


@dataclass
class SurrogatePrediction:
    """Prediction from the physics-informed surrogate for one process point."""

    power: float  # W
    velocity: float  # m/s
    beam_diameter: float  # m
    width: float  # m
    width_std: float  # m (GP posterior std)
    depth: float  # m
    depth_std: float  # m
    length: Optional[float]  # m (only if length model trained)
    length_std: Optional[float]
    rosenthal_width: float  # m
    rosenthal_depth: float  # m
    normalized_enthalpy: float
    mode: str  # 'conduction' | 'keyhole' | 'no_melt'
    depth_width_ratio: float


def conformal_calibration_factor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Compute a jackknife+ conformal calibration multiplier.

    Returns a scalar `c` such that the interval
        [y_pred - c * z * y_std, y_pred + c * z * y_std]
    (with z = 1.96) has approximately (1 - alpha) empirical coverage on the
    supplied calibration set. This is a variance-inflation calibration: the
    GP's posterior standard deviation is treated as a relative uncertainty
    profile and rescaled to achieve the target coverage.

    Mathematically, c is the (1 - alpha) empirical quantile of

        r_i = |y_true_i - y_pred_i| / (z * y_std_i)

    over the calibration points, so that at least (1 - alpha) of the
    standardized residuals fall inside the interval. This is a simplified
    (marginal) conformal calibration that is robust to the GP's variance
    being systematically too narrow or too wide.

    Args:
        y_true: measured values (n,).
        y_pred: point predictions (n,).
        y_std:  GP posterior standard deviations (n,).
        alpha:  significance level (default 0.05 -> 95% interval).

    Returns:
        The calibration multiplier c >= 0. If insufficient valid points,
        returns 1.0 (no inflation).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | (y_std <= 0))
    y_true, y_pred, y_std = y_true[mask], y_pred[mask], y_std[mask]
    if len(y_true) == 0:
        return 1.0

    z = 1.96
    residuals = np.abs(y_true - y_pred) / (z * y_std)
    q = float(np.quantile(residuals, 1.0 - alpha))
    # Guard against degeneracy / zero-variance points
    if not np.isfinite(q) or q <= 0:
        return 1.0
    return q


def empirical_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    levels: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute the empirical calibration (reliability) curve.

    For a set of nominal coverage levels (e.g. 0.5, 0.6, ..., 0.95), returns
    the fraction of true values falling inside the z-quantile prediction
    interval implied by each level. This is the standard reliability diagram
    used to assess whether a predictive distribution is calibrated across the
    whole quantile range, rather than at a single 95% level.

    Args:
        y_true: measured values (n,).
        y_pred: point predictions (n,).
        y_std:  predictive standard deviations (n,).
        levels: nominal coverage levels to evaluate (default 0.5..0.95).

    Returns:
        dict with 'levels' and 'coverage' arrays.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_std) | (y_std <= 0))
    y_true, y_pred, y_std = y_true[mask], y_pred[mask], y_std[mask]
    if levels is None:
        levels = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    out = []
    if len(y_true) == 0:
        return {"levels": np.asarray(levels), "coverage": np.full(len(levels), np.nan)}
    for lvl in levels:
        z = 1.96 * (lvl / 0.95)  # map nominal level to an effective z (approx)
        # More correct: use the normal quantile for the two-sided interval.
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + lvl / 2.0))
        lower = y_pred - z * y_std
        upper = y_pred + z * y_std
        out.append(float(np.mean((y_true >= lower) & (y_true <= upper))))
    return {"levels": np.asarray(levels), "coverage": np.asarray(out)}


class PhysicsInformedSurrogate:
    """Residual GP surrogate: y_pred = y_baseline + GP(physics features).

    Two independent GPs are trained: one for the width residual and one for
    the depth residual. A third GP is trained for the length residual when
    length data is available (Pramod IN718 only).

    The baseline (`y_baseline` above) defaults to Rosenthal, but can be
    swapped via two mutually-exclusive constructor arguments (if both are
    given, `goldak_mode_aware` takes precedence):
      - `goldak_shape`: a single global Goldak (a,b,c_front,c_rear) shape,
        from `goldak.calibrate_goldak_shape()` or
        `goldak.calibrate_goldak_shape_lsq()`.
      - `goldak_mode_aware`: a mode-aware calibration from
        `goldak.calibrate_goldak_mode_aware()`, which classifies
        conduction/keyhole mode from process parameters alone (via a
        fitted normalized-enthalpy threshold, not the measured
        depth/width ratio, which is unavailable at prediction time) and
        uses a separate Goldak shape per mode. This directly combines this
        project's two strongest depth-accuracy findings: the GP residual
        (data-driven local correction) on top of a baseline that already
        has the right depth/width aspect ratio for its predicted mode
        (physics-driven structural correction), rather than either alone.
    """

    def __init__(
        self,
        material: Material,
        absorptivity: float = 0.5,
        t0: float = 300.0,
        n_restarts: int = 5,
        random_seed: int = 42,
        goldak_shape: Optional[dict] = None,
        goldak_mode_aware: Optional[dict] = None,
    ) -> None:
        self.material = material
        self.absorptivity = absorptivity
        self.t0 = t0
        self.n_restarts = n_restarts
        self.random_seed = random_seed
        self.goldak_shape = goldak_shape
        self.goldak_mode_aware = goldak_mode_aware
        self.width_gp: Optional[GaussianProcessRegressor] = None
        self.depth_gp: Optional[GaussianProcessRegressor] = None
        self.length_gp: Optional[GaussianProcessRegressor] = None
        self.feature_names: list[str] = [
            "normalized_enthalpy",
            "peclet_number",
            "linear_energy_density",
            "areal_energy_density",
            "source_is_experimental",
        ]
        self._has_length_data = False
        # Conformal calibration multipliers (default 1.0 = no inflation)
        self.width_cal_factor: float = 1.0
        self.depth_cal_factor: float = 1.0

    # -- feature matrix -----------------------------------------------------

    def _feature_matrix(self, powers, velocities, beam_diameters, sources=None) -> np.ndarray:
        """Build the physics-feature matrix (n x 5) for a batch of cases.

        If `sources` is provided, appends a one-hot source-discrepancy
        indicator (1 for experimental data, 0 for FE-simulation) so the GP can
        learn experimental-vs-simulation bias. This addresses the confound of
        pooling measured and simulated melt-pool targets.
        """
        # The physics features come from physics_features(); the source
        # indicator is a synthetic column appended separately, so we must not
        # look it up in the feats dict.
        physics_names = [
            "normalized_enthalpy",
            "peclet_number",
            "linear_energy_density",
            "areal_energy_density",
        ]
        rows = []
        for i, (p, v, d) in enumerate(zip(powers, velocities, beam_diameters)):
            feats = physics_features(p, v, d, self.material, self.absorptivity, self.t0)
            row = [feats[name] for name in physics_names]
            if sources is not None:
                src = sources[i] if i < len(sources) else "unknown"
                is_exp = 1.0 if _is_experimental_source(src) else 0.0
                row.append(is_exp)
            rows.append(row)
        return np.asarray(rows, dtype=float)

    # -- Rosenthal baseline -------------------------------------------------

    def _rosenthal_baseline(self, power: float, velocity: float, beam_diameter: Optional[float] = None) -> Optional[dict]:
        """Evaluate the baseline model (Rosenthal by default, or Goldak --
        single-shape or mode-aware -- if configured via the constructor);
        None if no melt pool forms. Kept named `_rosenthal_baseline` for
        backward compatibility with existing call sites, even when it's
        actually evaluating Goldak -- the swap is controlled entirely by
        `self.goldak_mode_aware` / `self.goldak_shape`.

        `beam_diameter` is required (and must not be None) when
        `self.goldak_mode_aware` is set, since mode classification needs it
        to compute normalized enthalpy; Rosenthal and the single-shape
        Goldak baseline do not use it.
        """
        if self.goldak_mode_aware is not None:
            from .goldak import predict_mode_aware

            if beam_diameter is None:
                raise ValueError("beam_diameter is required when goldak_mode_aware is configured.")
            try:
                pred = predict_mode_aware(
                    power, velocity, beam_diameter, self.material, self.absorptivity, self.goldak_mode_aware, t0=self.t0
                )
                return {"width": pred["width"], "depth": pred["depth"]}
            except ValueError:
                return None

        if self.goldak_shape is not None:
            from .goldak import GoldakParameters, width_depth_fast

            params = GoldakParameters(
                power=power,
                velocity=velocity,
                absorptivity=self.absorptivity,
                t0=self.t0,
                a=self.goldak_shape["a"],
                b=self.goldak_shape["b"],
                c_front=self.goldak_shape["c_front"],
                c_rear=self.goldak_shape["c_rear"],
            )
            try:
                width, depth = width_depth_fast(params, self.material)
                return {"width": width, "depth": depth}
            except ValueError:
                return None

        params = ProcessParameters(power=power, velocity=velocity, absorptivity=self.absorptivity, t0=self.t0)
        try:
            return melt_pool_dimensions(params, self.material)
        except ValueError:
            return None

    # -- fitting ------------------------------------------------------------

    def fit(self, cases: list[SurrogateCase]) -> "PhysicsInformedSurrogate":
        """Fit the residual GPs on a list of SurrogateCase rows.

        Rows where the Rosenthal baseline fails (no melt pool) or where the
        measured geometry is missing are excluded from training.

        Cases are matched by the short alloy key (e.g. "316L", "IN718") used
        in the dataset, not the human-readable material name (e.g. "316L
        Stainless Steel"). We map the material's display name back to its
        MATERIALS key so the surrogate filters correctly.
        """
        from .materials import MATERIALS

        material_key = self.material.name
        for key, mat in MATERIALS.items():
            if mat == self.material:
                material_key = key
                break

        powers, velocities, beams, sources = [], [], [], []
        widths, depths, lengths = [], [], []
        ros_widths, ros_depths = [], []
        length_indices = []  # indices (into powers/velocities/beams) of cases with length data

        for c in cases:
            if c.alloy != material_key:
                continue
            baseline = self._rosenthal_baseline(c.power, c.velocity, c.beam_diameter)
            if baseline is None:
                continue
            if math.isnan(c.width) or math.isnan(c.depth):
                continue
            powers.append(c.power)
            velocities.append(c.velocity)
            beams.append(c.beam_diameter)
            sources.append(c.source)
            widths.append(c.width)
            depths.append(c.depth)
            ros_widths.append(baseline["width"])
            ros_depths.append(baseline["depth"])
            if not math.isnan(c.length):
                lengths.append(c.length)
                length_indices.append(len(powers) - 1)

        if len(powers) < 5:
            raise ValueError(f"Not enough training cases for {self.material.name}: {len(powers)}")

        X = self._feature_matrix(powers, velocities, beams, sources)

        # Width residual GP
        y_width_res = np.asarray(widths) - np.asarray(ros_widths)
        self.width_gp = GaussianProcessRegressor(n_restarts=self.n_restarts, random_seed=self.random_seed).fit(X, y_width_res)

        # Depth residual GP
        y_depth_res = np.asarray(depths) - np.asarray(ros_depths)
        self.depth_gp = GaussianProcessRegressor(n_restarts=self.n_restarts, random_seed=self.random_seed).fit(X, y_depth_res)

        # Length residual GP (only if length data present)
        if len(lengths) >= 5:
            self._has_length_data = True
            # Only use the subset of cases that actually have length measurements,
            # keeping the feature matrix and Rosenthal baseline aligned with the
            # length targets (Pramod IN718 reports length; other sources do not).
            X_len = self._feature_matrix(
                [powers[i] for i in length_indices],
                [velocities[i] for i in length_indices],
                [beams[i] for i in length_indices],
                sources=[sources[i] for i in length_indices],
            )
            y_len_res = np.asarray(lengths) - np.asarray([ros_widths[i] for i in length_indices])  # Rosenthal length is unreliable; use width as scale proxy
            self.length_gp = GaussianProcessRegressor(n_restarts=self.n_restarts, random_seed=self.random_seed).fit(X_len, y_len_res)

        return self

    # -- conformal calibration ---------------------------------------------

    def calibrate_conformal(
        self,
        cases: list[SurrogateCase],
        alpha: float = 0.05,
    ) -> "PhysicsInformedSurrogate":
        """Calibrate the prediction-interval multipliers on a calibration set.

        Uses the jackknife+ conformal method: for each calibration case, the
        absolute standardized residual |y_true - y_pred| / (z * y_std) is
        computed, and the (1 - alpha) quantile becomes the multiplier applied
        to the GP posterior standard deviation at prediction time. This
        guarantees approximately (1 - alpha) marginal coverage on the
        calibration distribution.

        Args:
            cases: SurrogateCase rows used for calibration (should be
                disjoint from the training set).
            alpha: significance level (default 0.05 -> 95% interval).

        Returns:
            self, with width_cal_factor and depth_cal_factor updated.
        """
        if self.width_gp is None or self.depth_gp is None:
            raise RuntimeError("Surrogate must be fitted before calibration.")

        from .materials import MATERIALS

        material_key = self.material.name
        for key, mat in MATERIALS.items():
            if mat == self.material:
                material_key = key
                break

        w_true, w_pred, w_std = [], [], []
        d_true, d_pred, d_std = [], [], []
        for c in cases:
            if c.alloy != material_key:
                continue
            pred = self.predict(c.power, c.velocity, c.beam_diameter)
            if math.isnan(pred.width) or math.isnan(pred.depth):
                continue
            w_true.append(c.width)
            w_pred.append(pred.width)
            w_std.append(pred.width_std)
            d_true.append(c.depth)
            d_pred.append(pred.depth)
            d_std.append(pred.depth_std)

        if len(w_true) > 0:
            self.width_cal_factor = conformal_calibration_factor(
                np.asarray(w_true), np.asarray(w_pred), np.asarray(w_std), alpha
            )
        if len(d_true) > 0:
            self.depth_cal_factor = conformal_calibration_factor(
                np.asarray(d_true), np.asarray(d_pred), np.asarray(d_std), alpha
            )
        return self

    # -- conformal calibration with disjoint split --------------------------

    def calibrate_conformal_split(
        self,
        calibration_cases: list[SurrogateCase],
        evaluation_cases: list[SurrogateCase],
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Calibrate on a disjoint set and report honest out-of-sample coverage.

        This fixes the circularity in :meth:`calibrate_conformal`, where the
        calibration factor and the reported coverage were computed on the same
        held-out fold (making coverage ~target by construction). Here the
        factor is fit strictly on `calibration_cases` and then evaluated on the
        *separate* `evaluation_cases`, so the reported coverage is a genuine
        out-of-sample statement.

        Args:
            calibration_cases: SurrogateCase rows used ONLY to fit the factor.
            evaluation_cases: SurrogateCase rows used ONLY to report coverage
                (must be disjoint from calibration_cases).
            alpha: significance level (default 0.05 -> 95% interval).

        Returns:
            dict with 'width' and 'depth' sub-dicts containing the calibration
            factor and the out-of-sample coverage on evaluation_cases.
        """
        if self.width_gp is None or self.depth_gp is None:
            raise RuntimeError("Surrogate must be fitted before calibration.")

        from .materials import MATERIALS

        material_key = self.material.name
        for key, mat in MATERIALS.items():
            if mat == self.material:
                material_key = key
                break

        # Exclude calibration cases that overlap the training set in source/alloy
        # to avoid leakage. Here we simply require the calibration cases to be a
        # strict subset of the SurrogateCase rows for this material.
        def _collect(cases):
            w_t, w_p, w_s, d_t, d_p, d_s = [], [], [], [], [], []
            for c in cases:
                if c.alloy != material_key:
                    continue
                pred = self.predict(c.power, c.velocity, c.beam_diameter)
                if math.isnan(pred.width) or math.isnan(pred.depth):
                    continue
                w_t.append(c.width)
                w_p.append(pred.width)
                w_s.append(pred.width_std)
                d_t.append(c.depth)
                d_p.append(pred.depth)
                d_s.append(pred.depth_std)
            return w_t, w_p, w_s, d_t, d_p, d_s

        # Fit factors on the calibration set only
        w_t, w_p, w_s, d_t, d_p, d_s = _collect(calibration_cases)
        if len(w_t) > 0:
            self.width_cal_factor = conformal_calibration_factor(np.asarray(w_t), np.asarray(w_p), np.asarray(w_s), alpha)
        if len(d_t) > 0:
            self.depth_cal_factor = conformal_calibration_factor(np.asarray(d_t), np.asarray(d_p), np.asarray(d_s), alpha)

        # Evaluate coverage on the disjoint evaluation set (honest OOS)
        w_t, w_p, w_s, d_t, d_p, d_s = _collect(evaluation_cases)
        w_cov = prediction_interval_coverage(np.asarray(w_t), np.asarray(w_p), np.asarray(w_s))
        d_cov = prediction_interval_coverage(np.asarray(d_t), np.asarray(d_p), np.asarray(d_s))
        return {
            "width": {
                "cal_factor": self.width_cal_factor,
                "coverage": w_cov["coverage"],
                "mean_interval_width": w_cov["mean_interval_width"],
                "n": w_cov["n"],
            },
            "depth": {
                "cal_factor": self.depth_cal_factor,
                "coverage": d_cov["coverage"],
                "mean_interval_width": d_cov["mean_interval_width"],
                "n": d_cov["n"],
            },
        }

    # -- prediction ---------------------------------------------------------

    def predict(self, power: float, velocity: float, beam_diameter: float) -> SurrogatePrediction:
        """Predict melt-pool geometry and uncertainty for one process point."""
        if self.width_gp is None or self.depth_gp is None:
            raise RuntimeError("Surrogate must be fitted before prediction.")

        baseline = self._rosenthal_baseline(power, velocity, beam_diameter)
        feats = physics_features(power, velocity, beam_diameter, self.material, self.absorptivity, self.t0)
        # Physical no-melt check: the Rosenthal point-source solution diverges at
        # the source and never returns None on its own, so we gate on the
        # normalized enthalpy -- below the melting-onset threshold there is no
        # melt pool regardless of the (singular) analytical solution.
        if baseline is None or feats["normalized_enthalpy"] < NO_MELT_ENTHALPY_THRESHOLD:
            return SurrogatePrediction(
                power=power,
                velocity=velocity,
                beam_diameter=beam_diameter,
                width=float("nan"),
                width_std=float("nan"),
                depth=float("nan"),
                depth_std=float("nan"),
                length=None,
                length_std=None,
                rosenthal_width=float("nan"),
                rosenthal_depth=float("nan"),
                normalized_enthalpy=feats["normalized_enthalpy"],
                mode="no_melt",
                depth_width_ratio=float("nan"),
            )

        # Prediction source: default to experimental (the surrogate is calibrated
        # on experimental targets; for FE-simulation targets pass source="Pramod2023").
        pred_source = getattr(self, "_prediction_source", "Hofmann2026")
        X = self._feature_matrix([power], [velocity], [beam_diameter], sources=[pred_source])

        w_res, w_std = self.width_gp.predict(X)
        d_res, d_std = self.depth_gp.predict(X)

        width = baseline["width"] + w_res[0]
        depth = baseline["depth"] + d_res[0]

        # Physics constraint: geometry must be positive
        width = max(width, 1e-9)
        depth = max(depth, 1e-9)

        length = None
        length_std = None
        if self._has_length_data and self.length_gp is not None:
            l_res, l_std = self.length_gp.predict(X)
            length = baseline["width"] + l_res[0]  # length model anchored to Rosenthal width scale
            length = max(length, 1e-9)
            length_std = l_std[0]

        feats = physics_features(power, velocity, beam_diameter, self.material, self.absorptivity, self.t0)
        dw_ratio = depth / width
        mode = classify_mode(dw_ratio, feats["normalized_enthalpy"])

        return SurrogatePrediction(
            power=power,
            velocity=velocity,
            beam_diameter=beam_diameter,
            width=width,
            width_std=w_std[0] * self.width_cal_factor,
            depth=depth,
            depth_std=d_std[0] * self.depth_cal_factor,
            length=length,
            length_std=length_std,
            rosenthal_width=baseline["width"],
            rosenthal_depth=baseline["depth"],
            normalized_enthalpy=feats["normalized_enthalpy"],
            mode=mode,
            depth_width_ratio=dw_ratio,
        )

    def predict_batch(
        self, powers: np.ndarray, velocities: np.ndarray, beam_diameters: np.ndarray
    ) -> list[SurrogatePrediction]:
        """Predict for arrays of process parameters (element-wise)."""
        return [
            self.predict(p, v, d)
            for p, v, d in zip(powers, velocities, beam_diameters)
        ]


class GlobalPhysicsInformedSurrogate:
    """Global multi-alloy residual GP surrogate.

    Unlike :class:`PhysicsInformedSurrogate` (which is refit per material),
    this surrogate trains a single set of GPs on ALL alloys at once. The
    feature matrix augments the dimensionless physics groups with the material
    thermophysical properties (k, rho, cp, t_melt, alpha), so the GP can
    interpolate across materials rather than extrapolate from one.

    The Rosenthal baseline is still computed per-case with the correct
    material properties, and the residual GP corrects it. This directly
    addresses the cross-alloy depth generalization limitation: the held-out
    alloy's properties are now *inputs* to the GP, not unseen labels.
    """

    def __init__(
        self,
        absorptivity: float = 0.5,
        t0: float = 300.0,
        n_restarts: int = 5,
        random_seed: int = 42,
    ) -> None:
        self.absorptivity = absorptivity
        self.t0 = t0
        self.n_restarts = n_restarts
        self.random_seed = random_seed
        self.width_gp: Optional[GaussianProcessRegressor] = None
        self.depth_gp: Optional[GaussianProcessRegressor] = None
        self.length_gp: Optional[GaussianProcessRegressor] = None
        self.feature_names: list[str] = [
            "normalized_enthalpy",
            "peclet_number",
            "linear_energy_density",
            "areal_energy_density",
            "k",
            "rho",
            "cp",
            "t_melt",
            "alpha",
            "source_is_experimental",
        ]
        self._has_length_data = False
        # Conformal calibration multipliers (default 1.0 = no inflation)
        self.width_cal_factor: float = 1.0
        self.depth_cal_factor: float = 1.0
        self._material_cache: dict[str, Material] = {}

    # -- helpers ------------------------------------------------------------

    def _get_material(self, alloy_key: str) -> Material:
        """Get the Material for a short dataset alloy key, cached."""
        if alloy_key not in self._material_cache:
            self._material_cache[alloy_key] = get_material(alloy_key)
        return self._material_cache[alloy_key]

    def _feature_matrix(self, powers, velocities, beam_diameters, alloys, sources=None) -> np.ndarray:
        """Build the augmented feature matrix (n x 10): physics + material props + source.

        If `sources` is provided, appends a one-hot source-discrepancy indicator
        (1 for experimental data, 0 for FE-simulation) so the GP can learn
        experimental-vs-simulation bias.
        """
        rows = []
        for i, (p, v, d, alloy) in enumerate(zip(powers, velocities, beam_diameters, alloys)):
            mat = self._get_material(alloy)
            feats = physics_features(p, v, d, mat, self.absorptivity, self.t0)
            row = [
                feats["normalized_enthalpy"],
                feats["peclet_number"],
                feats["linear_energy_density"],
                feats["areal_energy_density"],
                mat.k,
                mat.rho,
                mat.cp,
                mat.t_melt,
                mat.alpha,
            ]
            if sources is not None:
                src = sources[i] if i < len(sources) else "unknown"
                is_exp = 1.0 if _is_experimental_source(src) else 0.0
                row.append(is_exp)
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def _rosenthal_baseline(self, power: float, velocity: float, material: Material) -> Optional[dict]:
        """Evaluate the Rosenthal solution for a specific material."""
        params = ProcessParameters(power=power, velocity=velocity, absorptivity=self.absorptivity, t0=self.t0)
        try:
            return melt_pool_dimensions(params, material)
        except ValueError:
            return None

    # -- fitting ------------------------------------------------------------

    def fit(self, cases: list[SurrogateCase]) -> "GlobalPhysicsInformedSurrogate":
        """Fit the global residual GPs on all alloys at once."""
        powers, velocities, beams, alloys, sources = [], [], [], [], []
        widths, depths, lengths = [], [], []
        ros_widths, ros_depths = [], []
        length_indices = []

        for c in cases:
            mat = self._get_material(c.alloy)
            baseline = self._rosenthal_baseline(c.power, c.velocity, mat)
            if baseline is None:
                continue
            if math.isnan(c.width) or math.isnan(c.depth):
                continue
            powers.append(c.power)
            velocities.append(c.velocity)
            beams.append(c.beam_diameter)
            alloys.append(c.alloy)
            sources.append(c.source)
            widths.append(c.width)
            depths.append(c.depth)
            ros_widths.append(baseline["width"])
            ros_depths.append(baseline["depth"])
            if not math.isnan(c.length):
                lengths.append(c.length)
                length_indices.append(len(powers) - 1)

        if len(powers) < 5:
            raise ValueError(f"Not enough training cases: {len(powers)}")

        X = self._feature_matrix(powers, velocities, beams, alloys, sources)

        # Width residual GP
        y_width_res = np.asarray(widths) - np.asarray(ros_widths)
        self.width_gp = GaussianProcessRegressor(n_restarts=self.n_restarts, random_seed=self.random_seed).fit(X, y_width_res)

        # Depth residual GP
        y_depth_res = np.asarray(depths) - np.asarray(ros_depths)
        self.depth_gp = GaussianProcessRegressor(n_restarts=self.n_restarts, random_seed=self.random_seed).fit(X, y_depth_res)

        # Length residual GP (only if length data present)
        if len(lengths) >= 5:
            self._has_length_data = True
            X_len = self._feature_matrix(
                [powers[i] for i in length_indices],
                [velocities[i] for i in length_indices],
                [beams[i] for i in length_indices],
                [alloys[i] for i in length_indices],
                sources=[sources[i] for i in length_indices],
            )
            y_len_res = np.asarray(lengths) - np.asarray([ros_widths[i] for i in length_indices])
            self.length_gp = GaussianProcessRegressor(n_restarts=self.n_restarts, random_seed=self.random_seed).fit(X_len, y_len_res)

        return self

    # -- conformal calibration ---------------------------------------------

    def calibrate_conformal(
        self,
        cases: list[SurrogateCase],
        alpha: float = 0.05,
    ) -> "GlobalPhysicsInformedSurrogate":
        """Calibrate prediction-interval multipliers on a calibration set."""
        if self.width_gp is None or self.depth_gp is None:
            raise RuntimeError("Surrogate must be fitted before calibration.")

        w_true, w_pred, w_std = [], [], []
        d_true, d_pred, d_std = [], [], []
        for c in cases:
            pred = self.predict(c.power, c.velocity, c.beam_diameter, c.alloy)
            if math.isnan(pred.width) or math.isnan(pred.depth):
                continue
            w_true.append(c.width)
            w_pred.append(pred.width)
            w_std.append(pred.width_std)
            d_true.append(c.depth)
            d_pred.append(pred.depth)
            d_std.append(pred.depth_std)

        if len(w_true) > 0:
            self.width_cal_factor = conformal_calibration_factor(
                np.asarray(w_true), np.asarray(w_pred), np.asarray(w_std), alpha
            )
        if len(d_true) > 0:
            self.depth_cal_factor = conformal_calibration_factor(
                np.asarray(d_true), np.asarray(d_pred), np.asarray(d_std), alpha
            )
        return self

    # -- conformal calibration with disjoint split --------------------------

    def calibrate_conformal_split(
        self,
        calibration_cases: list[SurrogateCase],
        evaluation_cases: list[SurrogateCase],
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Calibrate on a disjoint set and report honest out-of-sample coverage.

        This fixes the circularity in :meth:`calibrate_conformal`, where the
        calibration factor and the reported coverage were computed on the same
        held-out fold (making coverage ~target by construction). Here the
        factor is fit strictly on `calibration_cases` and then evaluated on the
        *separate* `evaluation_cases`, so the reported coverage is a genuine
        out-of-sample statement.

        Args:
            calibration_cases: SurrogateCase rows used ONLY to fit the factor.
            evaluation_cases: SurrogateCase rows used ONLY to report coverage
                (must be disjoint from calibration_cases).
            alpha: significance level (default 0.05 -> 95% interval).

        Returns:
            dict with 'width' and 'depth' sub-dicts containing the calibration
            factor and the out-of-sample coverage on evaluation_cases.
        """
        if self.width_gp is None or self.depth_gp is None:
            raise RuntimeError("Surrogate must be fitted before calibration.")

        def _collect(cases):
            w_t, w_p, w_s, d_t, d_p, d_s = [], [], [], [], [], []
            for c in cases:
                pred = self.predict(c.power, c.velocity, c.beam_diameter, c.alloy)
                if math.isnan(pred.width) or math.isnan(pred.depth):
                    continue
                w_t.append(c.width)
                w_p.append(pred.width)
                w_s.append(pred.width_std)
                d_t.append(c.depth)
                d_p.append(pred.depth)
                d_s.append(pred.depth_std)
            return w_t, w_p, w_s, d_t, d_p, d_s

        # Fit factors on the calibration set only
        w_t, w_p, w_s, d_t, d_p, d_s = _collect(calibration_cases)
        if len(w_t) > 0:
            self.width_cal_factor = conformal_calibration_factor(np.asarray(w_t), np.asarray(w_p), np.asarray(w_s), alpha)
        if len(d_t) > 0:
            self.depth_cal_factor = conformal_calibration_factor(np.asarray(d_t), np.asarray(d_p), np.asarray(d_s), alpha)

        # Evaluate coverage on the disjoint evaluation set (honest OOS)
        w_t, w_p, w_s, d_t, d_p, d_s = _collect(evaluation_cases)
        w_cov = prediction_interval_coverage(np.asarray(w_t), np.asarray(w_p), np.asarray(w_s))
        d_cov = prediction_interval_coverage(np.asarray(d_t), np.asarray(d_p), np.asarray(d_s))
        return {
            "width": {
                "cal_factor": self.width_cal_factor,
                "coverage": w_cov["coverage"],
                "mean_interval_width": w_cov["mean_interval_width"],
                "n": w_cov["n"],
            },
            "depth": {
                "cal_factor": self.depth_cal_factor,
                "coverage": d_cov["coverage"],
                "mean_interval_width": d_cov["mean_interval_width"],
                "n": d_cov["n"],
            },
        }

    # -- prediction ---------------------------------------------------------

    def predict(self, power: float, velocity: float, beam_diameter: float, alloy: str) -> SurrogatePrediction:
        """Predict melt-pool geometry for one process point on a given alloy."""
        if self.width_gp is None or self.depth_gp is None:
            raise RuntimeError("Surrogate must be fitted before prediction.")

        mat = self._get_material(alloy)
        baseline = self._rosenthal_baseline(power, velocity, mat)
        feats = physics_features(power, velocity, beam_diameter, mat, self.absorptivity, self.t0)
        if baseline is None or feats["normalized_enthalpy"] < NO_MELT_ENTHALPY_THRESHOLD:
            return SurrogatePrediction(
                power=power,
                velocity=velocity,
                beam_diameter=beam_diameter,
                width=float("nan"),
                width_std=float("nan"),
                depth=float("nan"),
                depth_std=float("nan"),
                length=None,
                length_std=None,
                rosenthal_width=float("nan"),
                rosenthal_depth=float("nan"),
                normalized_enthalpy=feats["normalized_enthalpy"],
                mode="no_melt",
                depth_width_ratio=float("nan"),
            )

        # Prediction source: default to experimental (calibrated on experimental
        # targets; for FE-simulation targets pass source="Pramod2023").
        pred_source = getattr(self, "_prediction_source", "Hofmann2026")
        X = self._feature_matrix([power], [velocity], [beam_diameter], [alloy], sources=[pred_source])

        w_res, w_std = self.width_gp.predict(X)
        d_res, d_std = self.depth_gp.predict(X)

        width = max(baseline["width"] + w_res[0], 1e-9)
        depth = max(baseline["depth"] + d_res[0], 1e-9)

        length = None
        length_std = None
        if self._has_length_data and self.length_gp is not None:
            l_res, l_std = self.length_gp.predict(X)
            length = max(baseline["width"] + l_res[0], 1e-9)
            length_std = l_std[0]

        dw_ratio = depth / width
        mode = classify_mode(dw_ratio, feats["normalized_enthalpy"])

        return SurrogatePrediction(
            power=power,
            velocity=velocity,
            beam_diameter=beam_diameter,
            width=width,
            width_std=w_std[0] * self.width_cal_factor,
            depth=depth,
            depth_std=d_std[0] * self.depth_cal_factor,
            length=length,
            length_std=length_std,
            rosenthal_width=baseline["width"],
            rosenthal_depth=baseline["depth"],
            normalized_enthalpy=feats["normalized_enthalpy"],
            mode=mode,
            depth_width_ratio=dw_ratio,
        )

    def predict_batch(
        self,
        powers: np.ndarray,
        velocities: np.ndarray,
        beam_diameters: np.ndarray,
        alloys: list[str],
    ) -> list[SurrogatePrediction]:
        """Predict for arrays of process parameters (element-wise)."""
        return [
            self.predict(p, v, d, a)
            for p, v, d, a in zip(powers, velocities, beam_diameters, alloys)
        ]


# ---------------------------------------------------------------------------
# Physics-based mode classification
# ---------------------------------------------------------------------------

KEYHOLE_DW_RATIO_THRESHOLD = 0.8  # standard conduction/keyhole boundary

# Normalized enthalpy (dH/h_s) below which no melt pool forms. The Rosenthal
# point-source solution diverges at the source (1/R singularity) and therefore
# never returns None on its own, so we add an explicit physical no-melt check:
# dH/h_s < ~1 means the delivered energy is insufficient to reach T_melt.
NO_MELT_ENTHALPY_THRESHOLD = 1.0


def classify_mode(depth_width_ratio: float, normalized_enthalpy: Optional[float] = None) -> str:
    """Classify melt-pool mode from the depth/width ratio.

    Conduction mode: D/W < 0.8 (shallow, wide pool).
    Keyhole mode:    D/W >= 0.8 (deep, narrow pool with vapor depression).

    The normalized enthalpy is reported alongside for diagnostics but the
    classification is based on the geometric D/W ratio, which is the
    experimentally observable definition used across the L-PBF literature.
    """
    if math.isnan(depth_width_ratio):
        return "no_melt"
    if depth_width_ratio >= KEYHOLE_DW_RATIO_THRESHOLD:
        return "keyhole"
    return "conduction"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """R^2, RMSE, MAE for a set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan"), "n": 0}

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"r2": r2, "rmse": rmse, "mae": mae, "n": int(len(y_true))}


def bootstrap_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence intervals on R^2 and RMSE for a fixed prediction set.

    Resamples (y_true, y_pred) pairs with replacement `n_boot` times and
    reports the (alpha/2, 1-alpha/2) percentile interval on R^2 and RMSE.
    This does not retrain the model -- it quantifies how much the point
    estimate of a single held-out fold's metrics would vary under resampling
    of that fold's cases, which matters when n is small (as in a 2-4 alloy
    leave-one-alloy-out fold) and a single point R^2 could otherwise be
    over-interpreted.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n < 2:
        return {"r2_lo": float("nan"), "r2_hi": float("nan"), "rmse_lo": float("nan"), "rmse_hi": float("nan")}

    rng = np.random.default_rng(random_seed)
    r2s = np.empty(n_boot)
    rmses = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2s[b] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmses[b] = np.sqrt(np.mean((yt - yp) ** 2))

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "r2_lo": float(np.nanpercentile(r2s, lo)),
        "r2_hi": float(np.nanpercentile(r2s, hi)),
        "rmse_lo": float(np.nanpercentile(rmses, lo)),
        "rmse_hi": float(np.nanpercentile(rmses, hi)),
    }


def prediction_interval_coverage(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, z: float = 1.96) -> dict[str, float]:
    """Empirical coverage of the z-sigma prediction interval and mean width."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_std))
    y_true, y_pred, y_std = y_true[mask], y_pred[mask], y_std[mask]
    if len(y_true) == 0:
        return {"coverage": float("nan"), "mean_interval_width": float("nan"), "n": 0}

    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    mean_width = float(np.mean(2.0 * z * y_std))
    return {"coverage": coverage, "mean_interval_width": mean_width, "n": int(len(y_true))}
