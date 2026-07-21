"""Parameter sweeps over power x velocity, for building process maps.

A process map (width, depth, or depth/width ratio plotted as a function of laser
power and scan velocity) is the standard figure type for comparing an analytical
or numerical melt-pool model against a real process window. This module just
runs `melt_pool_dimensions` on a grid instead of a single point.
"""

import numpy as np

from .materials import Material
from .model import ProcessParameters, melt_pool_dimensions


def sweep(
    material: Material,
    powers: np.ndarray,
    velocities: np.ndarray,
    absorptivity: float,
    t0: float = 300.0,
) -> dict[str, np.ndarray]:
    """Evaluate melt-pool dimensions over a power x velocity grid.

    Args:
        material: Constant thermophysical properties.
        powers: 1D array of laser powers (W).
        velocities: 1D array of scan velocities (m/s).
        absorptivity: Fraction of power absorbed (held constant across the sweep).
        t0: Preheat / build-plate temperature (K).

    Returns:
        Dict with keys 'width', 'depth', 'length', 'aspect_ratio' (depth/width),
        each a 2D array of shape (len(velocities), len(powers)) in metres (aspect
        ratio is dimensionless). Points where no melt pool forms (sub-threshold
        parameters) are NaN.
    """
    shape = (len(velocities), len(powers))
    width = np.full(shape, np.nan)
    depth = np.full(shape, np.nan)
    length = np.full(shape, np.nan)

    for i, v in enumerate(velocities):
        for j, p in enumerate(powers):
            params = ProcessParameters(power=p, velocity=v, absorptivity=absorptivity, t0=t0)
            try:
                dims = melt_pool_dimensions(params, material)
            except ValueError:
                continue  # leave as NaN: no melt pool at this (power, velocity)
            width[i, j] = dims["width"]
            depth[i, j] = dims["depth"]
            length[i, j] = dims["length"]

    with np.errstate(invalid="ignore", divide="ignore"):
        aspect_ratio = depth / width

    return {"width": width, "depth": depth, "length": length, "aspect_ratio": aspect_ratio}
