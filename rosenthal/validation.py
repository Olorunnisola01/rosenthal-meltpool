"""Published single-track melt-pool measurements, for validating the model.

Data source: Zhang et al. (2024), "Understanding Melt Pool Behavior of 316L
Stainless Steel in Laser Powder Bed Fusion Additive Manufacturing," Micromachines,
15(2):170, DOI: 10.3390/mi15020170. Table 3, bare-plate single-track experiments.

The paper's own numerical model used an assumed laser absorptivity of 0.5 and a
100 um spot size; a powder layer thickness of 50 um is reported elsewhere in the
paper but does not apply to these particular bare-plate (no powder) cases. We use
absorptivity=0.5 to match the paper's own assumption, and t0=300 K (room
temperature is standard for bare-plate single-track experiments; the paper does
not state an explicit preheat, so this is the conventional default, not a value
taken from the paper).
"""

from dataclasses import dataclass

from scipy.optimize import brentq

from .materials import Material, get_material
from .model import ProcessParameters, melt_pool_dimensions


@dataclass(frozen=True)
class ValidationCase:
    """One published single-track measurement to compare a prediction against."""

    case_id: str
    material_name: str
    power: float  # W
    velocity: float  # m/s
    measured_width: float  # m
    measured_depth: float  # m
    source: str


ZHANG_2024_316L: list[ValidationCase] = [
    ValidationCase("N01", "316L", power=260.0, velocity=0.52, measured_width=114e-6, measured_depth=180e-6, source="Zhang et al. (2024), Table 3"),
    ValidationCase("N04", "316L", power=260.0, velocity=1.47, measured_width=94e-6, measured_depth=61e-6, source="Zhang et al. (2024), Table 3"),
    ValidationCase("N05", "316L", power=260.0, velocity=2.20, measured_width=83e-6, measured_depth=41e-6, source="Zhang et al. (2024), Table 3"),
    ValidationCase("N06", "316L", power=440.0, velocity=1.47, measured_width=98e-6, measured_depth=104e-6, source="Zhang et al. (2024), Table 3"),
]

ZHANG_2024_ABSORPTIVITY = 0.5
ZHANG_2024_T0 = 300.0


def compare_to_case(case: ValidationCase, absorptivity: float = ZHANG_2024_ABSORPTIVITY, t0: float = ZHANG_2024_T0) -> dict:
    """Run the model on one validation case and compute prediction error.

    Returns:
        Dict with predicted/measured width & depth (m), and percent error for
        each: positive means the model overpredicts, negative underpredicts.
    """
    material = get_material(case.material_name)
    params = ProcessParameters(power=case.power, velocity=case.velocity, absorptivity=absorptivity, t0=t0)
    predicted = melt_pool_dimensions(params, material)

    width_error_pct = (predicted["width"] - case.measured_width) / case.measured_width * 100
    depth_error_pct = (predicted["depth"] - case.measured_depth) / case.measured_depth * 100

    return {
        "case_id": case.case_id,
        "power": case.power,
        "velocity": case.velocity,
        "predicted_width": predicted["width"],
        "measured_width": case.measured_width,
        "width_error_pct": width_error_pct,
        "predicted_depth": predicted["depth"],
        "measured_depth": case.measured_depth,
        "depth_error_pct": depth_error_pct,
        "measured_aspect_ratio": case.measured_depth / case.measured_width,
    }


def calibrate_absorptivity(
    case: ValidationCase,
    target: str = "width",
    t0: float = ZHANG_2024_T0,
    bounds: tuple[float, float] = (0.01, 5.0),
) -> float | None:
    """Find the absorptivity that makes the predicted `target` dimension match
    the measured one exactly, searching (deliberately past the physical 0-1
    range, up to 5.0) to distinguish "needs a plausible absorptivity tweak"
    from "no absorptivity value, however extreme, can close this gap."

    Returns:
        The calibrated absorptivity, or None if no root exists in `bounds`
        (i.e. even absorptivity=500% under-predicts `target` -- a sign the
        point-source assumption itself, not the absorptivity guess, is the
        limiting factor for this case).
    """
    material = get_material(case.material_name)
    measured = case.measured_width if target == "width" else case.measured_depth

    def error(absorptivity: float) -> float:
        params = ProcessParameters(power=case.power, velocity=case.velocity, absorptivity=absorptivity, t0=t0)
        try:
            predicted = melt_pool_dimensions(params, material)[target]
        except ValueError:
            return -measured
        return predicted - measured

    try:
        return brentq(error, *bounds)
    except ValueError:
        return None
