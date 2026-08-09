"""Goldak double-ellipsoidal moving heat source solution for L-PBF melt-pool prediction.

Unlike the Rosenthal point-source solution (model.py), which is algebraically
forced to a semicircular cross-section (depth = width/2 always, see
model.melt_pool_dimensions docstring), the Goldak source has independently
adjustable width and depth shape parameters. This is the direct fix for that
structural limitation: it can represent conduction-mode pools (wider than
deep) and keyhole-mode pools (deeper than wide) as different parameter
regimes of the same model, rather than being unable to represent either.

References
----------
* Goldak, J., Chakravarti, A., & Bibby, M. (1984), "A new finite element
  model for welding heat sources," Metallurgical Transactions B, 15(2),
  299-305. Introduced the double-ellipsoidal Gaussian power-density model.
* Nguyen, N.T. et al. (1999), "Analytical solutions for transient
  temperature of semi-infinite body subjected to 3-D moving heat sources,"
  Welding Journal, 78, 265s-274s. First closed-form moving-source solution,
  correct only when the front and rear ellipsoids are equal (cf = cr).
* Fachinotti, V.D. & Cardona, A. (2008), "Semi-analytical solution of the
  thermal field induced by a moving double-ellipsoidal welding heat source
  in a semi-infinite body," Mecanica Computacional, XXVII, 1519-1530 (also
  published as Fachinotti, Anca & Cardona (2011), Communications in
  Numerical Methods in Engineering, 27(4), 595-607). Corrected and
  generalized Nguyen et al.'s solution to the true double-ellipsoidal case
  (cf != cr, ff != fr) by adding the missing error-function terms. This
  module implements that closed form (their eqs. 15-17) exactly, evaluated
  in the source-comoving quasi-steady limit (t -> infinity along the
  trajectory), analogous to how model.py's Rosenthal solution is the
  quasi-steady limit of the transient point-source Green's function.

Coordinate convention (matched to model.py, NOT to the Fachinotti & Cardona
paper's own x/y/z labels -- see the module-level note in `temperature()`):
    x: distance along the scan direction from the source, in the
       source-comoving frame (x > 0 ahead of the source, x < 0 behind it).
    y: transverse distance from the scan line (the width direction).
    z: depth below the surface, z >= 0.

By the symmetry noted in Fachinotti & Cardona's Remark II, this solution
already satisfies the insulated (adiabatic) free-surface boundary condition
at z=0 for z >= 0 without an extra image-source factor of 2 (contrast with
Rosenthal's point-source solution, which needs that doubling explicitly).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erf

from .materials import Material


@dataclass(frozen=True)
class GoldakParameters:
    """Laser/scan and double-ellipsoid shape parameters for one Goldak evaluation.

    Attributes:
        power: Laser power (W), before absorptivity is applied.
        velocity: Scan velocity (m/s).
        absorptivity: Fraction of `power` absorbed (see model.ProcessParameters).
        t0: Preheat / build-plate temperature (K).
        a: Half-width shape parameter of the ellipsoid (m) -- NOT the melt
            pool half-width itself; the two are related but not identical
            (the ellipsoid's Gaussian power density has 1/e^3 half-width a,
            the melt pool boundary is the T=T_melt isotherm of the resulting
            temperature field). Typically calibrated from data.
        b: Depth shape parameter of the ellipsoid (m), analogous to `a` but
            for the depth direction. This is the parameter that gives Goldak
            an adjustable aspect ratio unlike Rosenthal's fixed 0.5.
        c_front: Front-ellipsoid length parameter along the scan direction (m).
        c_rear: Rear-ellipsoid length parameter along the scan direction (m).
            Goldak's original default is c_rear = 4 * c_front (elongated
            trailing "comet tail"), but this is calibrated per-alloy here.
        f_front: Fraction of absorbed power deposited in the front ellipsoid.
            If None (default), computed from the continuity constraint
            f_front = 2*c_front/(c_front+c_rear) (Fachinotti & Cardona eq.
            18; Nguyen et al. 1999, 2004), which forces the power-density
            function to be continuous across the front/rear boundary plane.
            f_front + f_rear = 2 always (Goldak et al. 1984 convention).
    """

    power: float
    velocity: float
    absorptivity: float
    a: float
    b: float
    c_front: float
    c_rear: float
    t0: float = 300.0
    f_front: Optional[float] = None

    @property
    def absorbed_power(self) -> float:
        return self.absorptivity * self.power

    @property
    def f_front_value(self) -> float:
        if self.f_front is not None:
            return self.f_front
        return 2.0 * self.c_front / (self.c_front + self.c_rear)

    @property
    def f_rear_value(self) -> float:
        return 2.0 - self.f_front_value


def _integrand(tau: float, x: float, y: float, z: float, params: GoldakParameters, kappa: float) -> float:
    """Integrand of Fachinotti & Cardona (2008) eq. 15, in the tau = t - t'
    (elapsed time since deposition) substitution, evaluated in the
    source-comoving quasi-steady limit (see module docstring).

    `s = x + v*tau` is the position, relative to the *historical* source
    center at elapsed time tau in the past, of the fixed evaluation point in
    the comoving frame -- this is what the front/rear A_i, B_i terms (eqs.
    16-17) are evaluated at, in place of the paper's (z - v*t').
    """
    v = params.velocity
    a, b = params.a, params.b
    cf, cr = params.c_front, params.c_rear
    ff, fr = params.f_front_value, params.f_rear_value

    s = x + v * tau
    denom_a = 12.0 * kappa * tau + a * a
    denom_b = 12.0 * kappa * tau + b * b
    outer = math.exp(-3.0 * y * y / denom_a - 3.0 * z * z / denom_b) / (math.sqrt(denom_a) * math.sqrt(denom_b))

    sqrt_kt = math.sqrt(kappa * tau) if tau > 0 else 0.0

    def _a_i(ci: float) -> float:
        denom = 12.0 * kappa * tau + ci * ci
        return math.exp(-3.0 * s * s / denom) / math.sqrt(denom)

    def _b_i(ci: float) -> float:
        denom = 12.0 * kappa * tau + ci * ci
        if sqrt_kt == 0.0:
            # Limiting value as tau -> 0+: erf saturates to sign(s).
            return math.copysign(1.0, s) if s != 0.0 else 0.0
        return erf((ci / 2.0) * s / (sqrt_kt * math.sqrt(denom)))

    a_r, b_r = _a_i(cr), _b_i(cr)
    a_f, b_f = _a_i(cf), _b_i(cf)

    bracket = fr * a_r * (1.0 - b_r) + ff * a_f * (1.0 + b_f)
    return outer * bracket


def temperature(
    x: float,
    y: float,
    z: float,
    params: GoldakParameters,
    material: Material,
    tau_max: Optional[float] = None,
) -> float:
    """Quasi-steady-state temperature (K) at a point in the source-fixed frame.

    Args:
        x: Distance from the source along the scan direction (m); x > 0 ahead,
            x < 0 behind (matches model.py's Rosenthal convention).
        y: Transverse distance from the scan line (m).
        z: Depth below the surface (m), z >= 0.
        params: Laser/scan and ellipsoid shape parameters.
        material: Constant thermophysical properties.
        tau_max: Upper integration bound (s) for the elapsed-time integral, in
            place of a literal infinity. If None (default), computed from the
            problem's own length/diffusivity/velocity scales -- the
            integrand's characteristic width in tau is set by the smaller of
            (a) the diffusion time for 12*kappa*tau to become comparable to
            the ellipsoid semi-axes (~ci^2/(12*kappa)), and (b) the travel
            time for the comoving-frame offset s=x+v*tau to swing from
            strongly negative to strongly positive (~|x|/v plus a few
            multiples of the diffusion length over v). Using a generic fixed
            cutoff (e.g. a few seconds) silently fails for small ellipsoid
            parameters -- L-PBF melt pools have length scales ~1e-13 to
            1e-11 s^2/diffusivity, nine or more orders of magnitude below a
            fixed-seconds cutoff, so quadrature never samples the actual
            peak. Always let this default unless you have a specific reason
            to override it.

    Returns:
        Temperature in Kelvin.
    """
    kappa = material.alpha
    q = params.absorbed_power
    prefactor = (3.0 * math.sqrt(3.0) * q) / (material.rho * material.cp * math.pi**1.5)

    if tau_max is None:
        c_scale = max(params.a, params.b, params.c_front, params.c_rear)
        diffusion_time = (c_scale**2) / (12.0 * kappa)
        v = max(params.velocity, 1e-12)
        travel_time = (abs(x) + 10.0 * c_scale) / v
        tau_max = 50.0 * max(diffusion_time, travel_time)

    breakpoints = sorted(
        {
            (params.c_front**2) / (12.0 * kappa),
            (params.c_rear**2) / (12.0 * kappa),
            (params.a**2) / (12.0 * kappa),
            (params.b**2) / (12.0 * kappa),
        }
    )
    breakpoints = [bp for bp in breakpoints if 0.0 < bp < tau_max]

    integral, _ = quad(_integrand, 0.0, tau_max, args=(x, y, z, params, kappa), limit=400, points=breakpoints or None)
    return params.t0 + prefactor * integral


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(48)


def temperature_batch(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    params: GoldakParameters,
    material: Material,
    tau_max: Optional[float] = None,
) -> np.ndarray:
    """Vectorized quasi-steady temperature at many (x, y, z) points at once.

    Uses a fixed 48-point Gauss-Legendre quadrature (cubically warped toward
    tau=0, where the integrand's structure is concentrated -- see
    `temperature()`'s tau_max docstring) instead of adaptive `scipy.integrate.
    quad` per point. This trades a small, checked amount of accuracy
    (validated against the adaptive per-point `temperature()` in
    tests/test_physics_informed.py) for roughly two orders of magnitude more
    speed when evaluating many points -- needed to make nonlinear shape-
    parameter calibration (scipy.optimize.least_squares, which needs many
    function evaluations) computationally tractable. Do not use this for a
    single ad hoc evaluation where the adaptive `temperature()` is cheap
    enough and more conservatively accurate; use it for batch workloads
    (calibration, LOAO) where call count otherwise dominates runtime.

    Args:
        x, y, z: 1D arrays of equal length, evaluation points (m).
        params, material: as in `temperature()`.
        tau_max: as in `temperature()`, but computed once from `params`
            (not per-point) since it only depends on shape/velocity scales,
            not the individual x values -- uses the largest |x| in the batch
            plus the same margin `temperature()` uses.

    Returns:
        1D array of temperatures (K), same length as x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    kappa = material.alpha
    q = params.absorbed_power
    prefactor = (3.0 * math.sqrt(3.0) * q) / (material.rho * material.cp * math.pi**1.5)
    v = max(params.velocity, 1e-12)
    a, b, cf, cr = params.a, params.b, params.c_front, params.c_rear
    ff, fr = params.f_front_value, params.f_rear_value

    if tau_max is None:
        c_scale = max(a, b, cf, cr)
        diffusion_time = (c_scale**2) / (12.0 * kappa)
        travel_time = (float(np.max(np.abs(x))) + 10.0 * c_scale) / v
        tau_max = 50.0 * max(diffusion_time, travel_time)

    # Cubic warp: u in [0,1] -> tau = tau_max * u^3 concentrates quadrature
    # nodes near tau=0, where the integrand's characteristic structure
    # (governed by a,b,cf,cr, all << tau_max*v etc. for realistic melt pools)
    # actually lives.
    u = 0.5 * (_GL_NODES + 1.0)  # map [-1,1] -> [0,1]
    tau = tau_max * u**3  # (n_nodes,)
    dtau_du = 3.0 * tau_max * u**2
    weights = _GL_WEIGHTS * 0.5 * dtau_du  # d(tau) = dtau_du * du, du-weight = GL_WEIGHTS*0.5

    # Broadcast: (n_points, n_nodes)
    s = x[:, None] + v * tau[None, :]
    denom_a = 12.0 * kappa * tau[None, :] + a * a
    denom_b = 12.0 * kappa * tau[None, :] + b * b
    outer = np.exp(-3.0 * (y[:, None] ** 2) / denom_a - 3.0 * (z[:, None] ** 2) / denom_b) / (
        np.sqrt(denom_a) * np.sqrt(denom_b)
    )

    sqrt_kt = np.sqrt(np.maximum(kappa * tau[None, :], 0.0))

    def _a_i(ci):
        denom = 12.0 * kappa * tau[None, :] + ci * ci
        return np.exp(-3.0 * s * s / denom) / np.sqrt(denom)

    def _b_i(ci):
        denom = 12.0 * kappa * tau[None, :] + ci * ci
        with np.errstate(divide="ignore", invalid="ignore"):
            val = erf((ci / 2.0) * s / (sqrt_kt * np.sqrt(denom)))
        # tau=0 column: sqrt_kt=0 -> nan; use the sign(s) limiting value.
        zero_tau = sqrt_kt == 0.0
        if np.any(zero_tau):
            val = np.where(zero_tau, np.sign(s), val)
        return val

    a_r, b_r = _a_i(cr), _b_i(cr)
    a_f, b_f = _a_i(cf), _b_i(cf)
    bracket = fr * a_r * (1.0 - b_r) + ff * a_f * (1.0 + b_f)
    integrand = outer * bracket

    integral = np.sum(integrand * weights[None, :], axis=1)
    return params.t0 + prefactor * integral


def _temp_minus_melt(coord: float, axis: str, params: GoldakParameters, material: Material) -> float:
    if axis == "x":
        t = temperature(coord, 0.0, 0.0, params, material)
    elif axis == "y":
        t = temperature(0.0, coord, 0.0, params, material)
    elif axis == "z":
        t = temperature(0.0, 0.0, coord, params, material)
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
    return t - material.t_melt


def _find_isotherm_bound(axis: str, params: GoldakParameters, material: Material, search_max: float = 0.02) -> float:
    near = 1e-8
    if _temp_minus_melt(near, axis, params, material) < 0:
        raise ValueError(
            "No melt pool forms for these parameters: temperature is already below "
            "T_melt immediately adjacent to the source."
        )
    if _temp_minus_melt(search_max, axis, params, material) > 0:
        raise ValueError(
            f"Melt-pool boundary along {axis!r} exceeds the search window "
            f"(search_max={search_max} m). Increase search_max."
        )
    return brentq(_temp_minus_melt, near, search_max, args=(axis, params, material))


def melt_pool_dimensions(params: GoldakParameters, material: Material, search_max: float = 0.02) -> dict[str, float]:
    """Solve for melt-pool width, depth, and length from the T = T_melt isotherm.

    Unlike model.melt_pool_dimensions (Rosenthal), depth and half-width are
    governed by independent shape parameters (b and a respectively), so
    depth/width is NOT forced to any fixed ratio -- this is the direct fix
    for the Rosenthal model's structural depth=width/2 limitation.

    Returns:
        Dict with keys 'width', 'depth', 'length', 'length_front', 'length_back',
        all in metres.
    """
    half_width = _find_isotherm_bound("y", params, material, search_max)
    depth = _find_isotherm_bound("z", params, material, search_max)
    length_front = _find_isotherm_bound("x", params, material, search_max)

    def _temp_minus_melt_back(coord: float) -> float:
        return _temp_minus_melt(-coord, "x", params, material)

    near = 1e-8
    if _temp_minus_melt_back(near) < 0:
        raise ValueError("No melt pool forms behind the source for these parameters.")
    if _temp_minus_melt_back(search_max) > 0:
        raise ValueError(f"Trailing melt-pool extent exceeds search_max={search_max} m.")
    length_back = brentq(_temp_minus_melt_back, near, search_max)

    return {
        "width": 2 * half_width,
        "depth": depth,
        "length": length_front + length_back,
        "length_front": length_front,
        "length_back": length_back,
    }


def width_depth_fast(params: GoldakParameters, material: Material, search_max: float = 0.02) -> tuple[float, float]:
    """Fast width and depth only, via `temperature_batch`'s fixed-node
    quadrature instead of adaptive `scipy.integrate.quad`. Skips length
    (not needed for calibration against width/depth data). Raises
    ValueError if no melt pool forms. See `temperature_batch`'s docstring
    for the accuracy/speed tradeoff -- validated to agree with the adaptive
    `melt_pool_dimensions()` to within ~0.01% in
    tests/test_physics_informed.py.
    """

    def _f_y(coord: float) -> float:
        return float(temperature_batch(np.array([0.0]), np.array([coord]), np.array([0.0]), params, material)[0]) - material.t_melt

    def _f_z(coord: float) -> float:
        return float(temperature_batch(np.array([0.0]), np.array([0.0]), np.array([coord]), params, material)[0]) - material.t_melt

    near = 1e-8
    if _f_y(near) < 0 or _f_z(near) < 0:
        raise ValueError("No melt pool forms for these parameters.")
    if _f_y(search_max) > 0 or _f_z(search_max) > 0:
        raise ValueError(f"Melt-pool boundary exceeds search_max={search_max} m.")

    half_width = brentq(_f_y, near, search_max)
    depth = brentq(_f_z, near, search_max)
    return 2.0 * half_width, depth


def calibrate_goldak_shape(cases: list, material: Material, absorptivity: float) -> dict[str, float]:
    """Calibrate Goldak ellipsoid shape parameters from measured melt-pool data.

    Method: direct geometric initialization from measured melt-pool geometry
    -- a0 = mean(measured half-width), b0 = mean(measured depth) -- with the
    front/rear length split set from Goldak et al. (1984)'s literature-
    default ratio c_rear = 4*c_front (their recommended default "in absence
    of better data," which applies here since none of the 4 alloys in this
    dataset report melt-pool *length*, only width and depth), and
    c_front0 = a0 (a common simplification used when no length data is
    available to calibrate against; see the FEA calibration studies cited by
    Fachinotti & Cardona).

    This is a standard, established, one-step calibration recipe (not a
    globally optimal nonlinear least-squares fit -- report it as exactly
    that in the manuscript). A refinement step (searching a uniform scale
    factor against predicted width) was evaluated and rejected: enlarging
    a, b, c_front, c_rear together dilutes the same absorbed power over a
    larger volume, which drops peak temperature below melting non-
    monotonically as scale increases -- this makes an ordinary root-search
    on scale unreliable (predicted width vs. scale is not monotonic, and
    frequently no melt pool forms at all for a plausible scale range). The
    direct estimate below is therefore used as the shape-parameter
    calibration, and its resulting width/depth fit quality is reported
    empirically via the ablation/LOAO metrics (scripts/train_physics_informed.py)
    rather than assumed.

    Args:
        cases: SurrogateCase rows for a single alloy (already filtered).
        material: Material properties.
        absorptivity: Absorptivity used elsewhere for this alloy (kept as an
            argument for API symmetry with calibrate_absorptivity(), though
            this shape calibration does not itself evaluate the Goldak model).

    Returns:
        Dict with keys "a", "b", "c_front", "c_rear", "n_cases".
    """
    import math as _math

    import numpy as _np

    valid = [c for c in cases if not (_math.isnan(c.width) or _math.isnan(c.depth))]
    if not valid:
        raise ValueError("No valid cases (with measured width and depth) to calibrate against.")

    a0 = float(_np.mean([c.width for c in valid])) / 2.0
    b0 = float(_np.mean([c.depth for c in valid]))

    return {
        "a": a0,
        "b": b0,
        "c_front": a0,
        "c_rear": 4.0 * a0,
        "n_cases": len(valid),
    }


def calibrate_goldak_depth_only(
    cases: list,
    material: Material,
    absorptivity: float,
    n_calibration_cases: int = 18,
) -> dict[str, float]:
    """Calibrate ONLY the depth shape parameter b, holding a (and c_front,
    c_rear) fixed at the one-step geometric estimate from measured width.

    Rationale: the joint (a, b) least-squares fit in
    `calibrate_goldak_shape_lsq()` trades width accuracy off against depth
    accuracy, because both are fit to the same nondimensionalized objective
    -- in testing this made width *worse* than Rosenthal in every alloy
    (even where depth improved), and produced an unstable, physically
    implausible fit for IN718 (predicted depth/width ratio 5.8). Since the
    diagnosed problem is specifically that Rosenthal cannot represent depth
    independently of width (the depth=width/2 structural limitation, see
    model.py), the more targeted and stable fix is to leave `a` (and
    therefore width behavior) alone and calibrate only `b` against measured
    depth. This is a 1D optimization -- much more robust than the 2D case,
    and directly answers the question this whole exercise was about
    ("depth done the right way") without collaterally breaking width.

    Returns:
        Dict with keys "a", "b", "c_front", "c_rear", "n_cases",
        "n_calibration_cases", "success", "cost".
    """
    from scipy.optimize import least_squares

    valid = [c for c in cases if not (math.isnan(c.width) or math.isnan(c.depth))]
    if not valid:
        raise ValueError("No valid cases (with measured width and depth) to calibrate against.")

    order = sorted(valid, key=lambda c: c.width)
    idx = np.linspace(0, len(order) - 1, min(n_calibration_cases, len(order))).astype(int)
    subset = [order[i] for i in sorted(set(idx))]

    init = calibrate_goldak_shape(valid, material, absorptivity)
    a_fixed = init["a"]
    b0 = init["b"]

    depths_true = np.array([c.depth for c in subset])
    mean_depth = float(np.mean(depths_true))

    def _residuals(theta: np.ndarray) -> np.ndarray:
        (b,) = theta
        b = max(b, 1e-7)
        depths_pred = np.full(len(subset), np.nan)
        for i, c in enumerate(subset):
            params = GoldakParameters(
                power=c.power,
                velocity=c.velocity,
                absorptivity=absorptivity,
                a=a_fixed,
                b=b,
                c_front=a_fixed,
                c_rear=4.0 * a_fixed,
            )
            try:
                _, d = width_depth_fast(params, material)
            except ValueError:
                d = 0.0
            depths_pred[i] = d
        return (depths_pred - depths_true) / mean_depth

    result = least_squares(
        _residuals,
        x0=[b0],
        bounds=([b0 * 0.05], [b0 * 10.0]),
        xtol=1e-4,
        ftol=1e-4,
        max_nfev=40,
    )
    (b_fit,) = result.x

    return {
        "a": float(a_fixed),
        "b": float(b_fit),
        "c_front": float(a_fixed),
        "c_rear": float(4.0 * a_fixed),
        "n_cases": len(valid),
        "n_calibration_cases": len(subset),
        "success": bool(result.success),
        "cost": float(result.cost),
    }


def calibrate_goldak_shape_lsq(
    cases: list,
    material: Material,
    absorptivity: float,
    n_calibration_cases: int = 18,
    random_seed: int = 42,
    width_weight: float = 1.0,
) -> dict[str, float]:
    """Nonlinear least-squares calibration of (a, b), the Goldak width/depth
    shape parameters, against real per-case measured width AND depth (not
    just the mean width used by the one-step `calibrate_goldak_shape()`).

    This is the proper fit `calibrate_goldak_shape()` explicitly deferred
    (see that function's docstring) -- made tractable here by
    `width_depth_fast()` / `temperature_batch()`'s fixed-quadrature speedup
    (~25x faster than the adaptive per-point solver), which brings a
    per-case evaluation down to ~25-50 ms, so a `scipy.optimize.least_squares`
    run over a representative subset of cases completes in well under a
    minute instead of hours.

    c_front and c_rear are NOT fit independently -- they are kept coupled to
    a via Goldak's literature-default c_rear = 4*c_front, c_front = a
    (matching `calibrate_goldak_shape()`'s convention), since none of the 4
    alloys in this dataset report melt-pool *length* to fit c_front/c_rear
    against. Only (a, b) -- which directly control the width/depth aspect
    ratio this calibration exists to fix -- are optimized.

    Args:
        cases: SurrogateCase rows for a single alloy (already filtered).
        material: Material properties.
        absorptivity: Absorptivity to use for all Goldak evaluations.
        n_calibration_cases: number of cases (evenly spread by measured
            width) used as the least-squares fitting set.
        random_seed: unused directly (subset selection is deterministic,
            evenly spread by width) but kept for API consistency/future use.

    Returns:
        Dict with keys "a", "b", "c_front", "c_rear", "n_cases",
        "success" (bool), "cost" (final least-squares cost).
    """
    from scipy.optimize import least_squares

    valid = [c for c in cases if not (math.isnan(c.width) or math.isnan(c.depth))]
    if not valid:
        raise ValueError("No valid cases (with measured width and depth) to calibrate against.")

    order = sorted(valid, key=lambda c: c.width)
    idx = np.linspace(0, len(order) - 1, min(n_calibration_cases, len(order))).astype(int)
    subset = [order[i] for i in sorted(set(idx))]

    init = calibrate_goldak_shape(valid, material, absorptivity)
    a0, b0 = init["a"], init["b"]

    widths_true = np.array([c.width for c in subset])
    depths_true = np.array([c.depth for c in subset])

    def _residuals(theta: np.ndarray) -> np.ndarray:
        a, b = theta
        a = max(a, 1e-7)
        b = max(b, 1e-7)
        widths_pred = np.full(len(subset), np.nan)
        depths_pred = np.full(len(subset), np.nan)
        for i, c in enumerate(subset):
            params = GoldakParameters(
                power=c.power,
                velocity=c.velocity,
                absorptivity=absorptivity,
                a=a,
                b=b,
                c_front=a,
                c_rear=4.0 * a,
            )
            try:
                w, d = width_depth_fast(params, material)
            except ValueError:
                # Penalize (rather than crash) parameter regions with no melt,
                # steering the optimizer back toward feasible regions.
                w, d = 0.0, 0.0
            widths_pred[i] = w
            depths_pred[i] = d
        # Nondimensionalize residuals by the mean measured scale so width
        # and depth contribute comparably to the cost despite different
        # magnitudes.
        return np.concatenate(
            [
                width_weight * (widths_pred - widths_true) / np.mean(widths_true),
                (depths_pred - depths_true) / np.mean(depths_true),
            ]
        )

    result = least_squares(
        _residuals,
        x0=[a0, b0],
        bounds=([a0 * 0.1, b0 * 0.1], [a0 * 5.0, b0 * 5.0]),
        xtol=1e-4,
        ftol=1e-4,
        max_nfev=60,
    )
    a_fit, b_fit = result.x

    return {
        "a": float(a_fit),
        "b": float(b_fit),
        "c_front": float(a_fit),
        "c_rear": float(4.0 * a_fit),
        "n_cases": len(valid),
        "n_calibration_cases": len(subset),
        "success": bool(result.success),
        "cost": float(result.cost),
    }


def fit_mode_threshold(cases: list, material: Material, absorptivity: float, n_grid: int = 30) -> dict[str, float]:
    """Fit a normalized-enthalpy threshold that separates conduction- from
    keyhole-mode cases, usable at prediction time (unlike the measured
    depth/width ratio, which is only known after the fact).

    Ground truth mode labels come from each case's measured depth/width
    ratio against `physics_informed.KEYHOLE_DW_RATIO_THRESHOLD` (the
    standard geometric definition used throughout this project and the
    wider L-PBF literature). The threshold on normalized enthalpy -- a
    process-parameter-only quantity, computable before any measurement --
    is chosen by grid search over its training-data percentiles to
    maximize agreement with those ground-truth labels. This is the
    established approach referenced in King et al. (2014)'s normalized-
    enthalpy keyhole-transition criterion, applied here as a per-alloy
    calibrated threshold rather than a universal constant.

    Returns:
        Dict with keys "threshold" (normalized enthalpy) and
        "train_accuracy" (fraction of training cases correctly classified
        at that threshold).
    """
    from .physics_informed import KEYHOLE_DW_RATIO_THRESHOLD, physics_features

    valid = [c for c in cases if not (math.isnan(c.width) or math.isnan(c.depth)) and c.width > 0]
    if not valid:
        raise ValueError("No valid cases to fit a mode threshold against.")

    enth = np.array(
        [
            physics_features(c.power, c.velocity, c.beam_diameter, material, absorptivity, 300.0)[
                "normalized_enthalpy"
            ]
            for c in valid
        ]
    )
    labels = np.array([1 if (c.depth / c.width) >= KEYHOLE_DW_RATIO_THRESHOLD else 0 for c in valid])

    best_thr, best_acc = float(enth[0]), -1.0
    for thr in np.percentile(enth, np.linspace(5, 95, n_grid)):
        pred = (enth >= thr).astype(int)
        acc = float(np.mean(pred == labels))
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)

    return {"threshold": best_thr, "train_accuracy": best_acc}


def calibrate_goldak_mode_aware(
    cases: list,
    material: Material,
    absorptivity: float,
    min_cases_per_mode: int = 8,
) -> dict:
    """Full mode-aware Goldak calibration: a normalized-enthalpy threshold
    (usable at prediction time, via `fit_mode_threshold()`) plus separate
    (a, b) shape parameters for conduction- and keyhole-mode cases, each fit
    via `calibrate_goldak_shape_lsq()` on its own mode's data.

    Falls back to a single global shape (from `calibrate_goldak_shape_lsq()`
    on ALL cases, used for both modes) if either mode has fewer than
    `min_cases_per_mode` cases -- a per-mode fit on too few points is not
    trustworthy, and this is reported via `"mode_aware": False` in the
    result rather than silently returning an overfit per-mode estimate.

    Returns:
        Dict with keys "mode_aware" (bool), "threshold", "train_accuracy",
        "shape_conduction", "shape_keyhole" (each a dict as returned by
        `calibrate_goldak_shape_lsq()`; identical to each other if
        `mode_aware` is False), "n_conduction", "n_keyhole".
    """
    from .physics_informed import KEYHOLE_DW_RATIO_THRESHOLD

    valid = [c for c in cases if not (math.isnan(c.width) or math.isnan(c.depth)) and c.width > 0]
    if not valid:
        raise ValueError("No valid cases to calibrate against.")

    ratios = [c.depth / c.width for c in valid]
    conduction = [c for c, r in zip(valid, ratios) if r < KEYHOLE_DW_RATIO_THRESHOLD]
    keyhole = [c for c, r in zip(valid, ratios) if r >= KEYHOLE_DW_RATIO_THRESHOLD]

    thr_result = fit_mode_threshold(valid, material, absorptivity)

    if len(conduction) < min_cases_per_mode or len(keyhole) < min_cases_per_mode:
        global_shape = calibrate_goldak_shape_lsq(valid, material, absorptivity=absorptivity)
        return {
            "mode_aware": False,
            "threshold": thr_result["threshold"],
            "train_accuracy": thr_result["train_accuracy"],
            "shape_conduction": global_shape,
            "shape_keyhole": global_shape,
            "n_conduction": len(conduction),
            "n_keyhole": len(keyhole),
        }

    shape_conduction = calibrate_goldak_shape_lsq(
        conduction, material, absorptivity=absorptivity, n_calibration_cases=min(15, len(conduction))
    )
    shape_keyhole = calibrate_goldak_shape_lsq(
        keyhole, material, absorptivity=absorptivity, n_calibration_cases=min(15, len(keyhole))
    )

    return {
        "mode_aware": True,
        "threshold": thr_result["threshold"],
        "train_accuracy": thr_result["train_accuracy"],
        "shape_conduction": shape_conduction,
        "shape_keyhole": shape_keyhole,
        "n_conduction": len(conduction),
        "n_keyhole": len(keyhole),
    }


def predict_mode_aware(
    power: float,
    velocity: float,
    beam_diameter: float,
    material: Material,
    absorptivity: float,
    calibration: dict,
    t0: float = 300.0,
) -> dict[str, float]:
    """Predict width and depth using a mode-aware Goldak calibration from
    `calibrate_goldak_mode_aware()`, classifying mode from process
    parameters alone (normalized enthalpy vs. the fitted threshold) -- not
    from measured geometry, which is unavailable at prediction time. This
    is the actually-deployable form of the mode-stratified correction;
    evaluating the mode-specific shapes against oracle (measured) mode
    labels, as the calibration/diagnostic scripts do, would overstate
    real-world performance.

    Returns:
        Dict with keys "width", "depth", "mode" ("conduction" or
        "keyhole"). Raises ValueError if no melt pool forms under the
        selected mode's shape parameters.
    """
    from .physics_informed import physics_features

    feats = physics_features(power, velocity, beam_diameter, material, absorptivity, t0)
    mode = "keyhole" if feats["normalized_enthalpy"] >= calibration["threshold"] else "conduction"
    shape = calibration["shape_keyhole"] if mode == "keyhole" else calibration["shape_conduction"]

    params = GoldakParameters(
        power=power,
        velocity=velocity,
        absorptivity=absorptivity,
        t0=t0,
        a=shape["a"],
        b=shape["b"],
        c_front=shape["c_front"],
        c_rear=shape["c_rear"],
    )
    width, depth = width_depth_fast(params, material)
    return {"width": width, "depth": depth, "mode": mode}
