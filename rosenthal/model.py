"""Rosenthal moving point-heat-source solution for L-PBF melt-pool prediction.

Reference: D. Rosenthal, "The theory of moving sources of heat and its application
to metal treatments," Trans. ASME, 68, 849-866 (1946).

The model gives the quasi-steady-state temperature field around a point heat source
moving at constant velocity v over/through a semi-infinite solid, in a reference
frame that travels with the source:

    T(x, y, z) - T0 = (Q / (2 * pi * k * R)) * exp(-v * (R + x) / (2 * alpha))

where x is the coordinate behind the source (source moves in the +x direction) and
R = sqrt(x^2 + y^2 + z^2) is the radial distance from the source.

This is a conduction-mode solution: it assumes all absorbed power is deposited at a
single point on (or just below) the surface and diffuses purely by conduction. It
does not capture keyhole-mode vaporization/recoil pressure effects, so predictions
in the keyhole regime (typically high power, low speed, deep narrow pools) should be
treated as qualitative only.
"""

import math
from dataclasses import dataclass

from scipy.optimize import brentq

from .materials import Material


@dataclass(frozen=True)
class ProcessParameters:
    """Laser/scan parameters for one Rosenthal evaluation.

    Attributes:
        power: Laser power (W), before absorptivity is applied.
        velocity: Scan velocity (m/s).
        absorptivity: Fraction of `power` actually absorbed into the melt pool
            (material- and wavelength-dependent; typical L-PBF values are 0.3-0.7
            for as-deposited powder beds). Absorbed power Q = absorptivity * power.
        t0: Preheat / build-plate temperature (K).
    """

    power: float
    velocity: float
    absorptivity: float
    t0: float = 300.0

    @property
    def absorbed_power(self) -> float:
        return self.absorptivity * self.power


def temperature(x: float, y: float, z: float, params: ProcessParameters, material: Material) -> float:
    """Quasi-steady-state temperature (K) at a point in the source-fixed frame.

    Args:
        x: Distance behind the source along the scan direction (m). x < 0 is
            behind the source (in its wake), x > 0 is ahead of it.
        y: Transverse distance from the scan line (m).
        z: Depth below the surface (m), positive into the material.
        params: Laser/scan process parameters.
        material: Constant thermophysical properties.

    Returns:
        Temperature in Kelvin. Diverges at R -> 0 (at the source itself), as
        expected for a point-source idealization.
    """
    r = (x**2 + y**2 + z**2) ** 0.5
    if r == 0:
        raise ValueError("Rosenthal solution is singular at the source point (R=0).")

    q = params.absorbed_power
    k = material.k
    alpha = material.alpha
    v = params.velocity

    return params.t0 + (q / (2 * math.pi * k * r)) * math.exp(-v * (r + x) / (2 * alpha))


def _temp_minus_melt(coord: float, axis: str, params: ProcessParameters, material: Material) -> float:
    """T(...) - T_melt along one principal axis, for use as a root-finding objective."""
    if axis == "x":
        t = temperature(coord, 0.0, 0.0, params, material)
    elif axis == "y":
        t = temperature(0.0, coord, 0.0, params, material)
    elif axis == "z":
        t = temperature(0.0, 0.0, coord, params, material)
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
    return t - material.t_melt


def _find_isotherm_bound(axis: str, params: ProcessParameters, material: Material, search_max: float = 0.02) -> float:
    """Find the positive distance along `axis` where T drops to T_melt.

    Uses bisection (via scipy.optimize.brentq) between a point very close to the
    source (still above melting) and `search_max` (expected to be below melting).
    Raises ValueError if no melt pool forms at all (T never reaches T_melt even at
    the source), i.e. these process parameters are sub-threshold.
    """
    near = 1e-7  # 0.1 micron from the source; T is enormous here for any real Q
    if _temp_minus_melt(near, axis, params, material) < 0:
        raise ValueError(
            "No melt pool forms for these parameters: temperature is already below "
            "T_melt immediately adjacent to the source. Increase power, reduce "
            "velocity, or check absorptivity."
        )
    if _temp_minus_melt(search_max, axis, params, material) > 0:
        raise ValueError(
            f"Melt-pool boundary along {axis!r} exceeds the search window "
            f"(search_max={search_max} m). Increase search_max."
        )
    return brentq(_temp_minus_melt, near, search_max, args=(axis, params, material))


def melt_pool_dimensions(params: ProcessParameters, material: Material, search_max: float = 0.02) -> dict[str, float]:
    """Solve for melt-pool width, depth, and length from the T = T_melt isotherm.

    Width and depth are found on the plane through the source (x=0), where the
    pool is widest/deepest. Length is the sum of the leading (ahead of the source,
    x>0) and trailing (behind the source, x<0) extents along the scan line.

    Structural limitation -- depth/width is always exactly 0.5: at x=0, both
    the half-width equation (solved along y) and the depth equation (solved
    along z) reduce to the identical function of a single radial distance,
    since R = sqrt(y^2 + z^2) at x=0 treats y and z interchangeably. This
    means half_width == depth for *any* power, velocity, absorptivity, or
    material -- the cross-section is a perfect semicircle by construction.
    This is an exact, provable property of the canonical point-source-at-
    surface Rosenthal formula, not an approximation or numerical artifact.
    Real melt pools are not semicircular (conduction-mode pools are wider
    than deep; keyhole-mode pools are deeper than wide) -- this model cannot
    represent that variation at all. Reproducing a non-0.5 aspect ratio
    requires a different heat-source model (e.g. Goldak's double-ellipsoidal
    source), not a parameter change to this one.

    Caveat on `length` / `length_back`: exactly on the trailing centerline
    (y=0, z=0, x<0), R = -x, so R+x = 0 for every point on that line -- the
    exponential decay term is identically 1 there, and temperature falls off only
    as 1/R rather than exponentially. This is a real feature of the idealized
    point-source solution (not a numerical bug), and it means `length_back` (and
    therefore `length`) tends to be substantially overpredicted relative to real
    single-track measurements, which use a distributed (e.g. Gaussian) beam rather
    than a true point. Treat `width` and `depth` as the more reliable outputs;
    treat `length` as qualitative.

    Returns:
        Dict with keys 'width', 'depth', 'length', 'length_front', 'length_back',
        all in metres.
    """
    half_width = _find_isotherm_bound("y", params, material, search_max)
    depth = _find_isotherm_bound("z", params, material, search_max)
    length_front = _find_isotherm_bound("x", params, material, search_max)

    # Behind the source, T(x) - T_melt is evaluated with x negative; mirror the
    # search by solving on -x and negating.
    def _temp_minus_melt_back(coord: float) -> float:
        return _temp_minus_melt(-coord, "x", params, material)

    near = 1e-7
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
