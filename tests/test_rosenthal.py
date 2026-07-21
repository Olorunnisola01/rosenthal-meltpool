import math

import pytest

from rosenthal import MATERIALS, ProcessParameters, get_material, melt_pool_dimensions, temperature


@pytest.mark.parametrize("name", list(MATERIALS))
def test_reasonable_melt_pool_for_typical_lpbf_parameters(name):
    """Typical L-PBF power/speed should produce a melt pool of plausible size."""
    material = get_material(name)
    params = ProcessParameters(power=200.0, velocity=0.8, absorptivity=0.4, t0=300.0)

    dims = melt_pool_dimensions(params, material)

    # Real L-PBF single tracks are tens to a few hundred microns in each dimension.
    for key in ("width", "depth", "length"):
        assert 1e-6 < dims[key] < 2e-3, f"{name} {key}={dims[key]} out of plausible range"

    assert dims["length"] == pytest.approx(dims["length_front"] + dims["length_back"])


def test_temperature_decreases_with_distance_from_source():
    material = get_material("316L")
    params = ProcessParameters(power=200.0, velocity=0.8, absorptivity=0.4)

    t_near = temperature(-1e-5, 0.0, 0.0, params, material)
    t_far = temperature(-5e-3, 0.0, 0.0, params, material)

    assert t_near > t_far


def test_temperature_singular_at_source():
    material = get_material("Ti-6Al-4V")
    params = ProcessParameters(power=200.0, velocity=0.8, absorptivity=0.4)

    with pytest.raises(ValueError):
        temperature(0.0, 0.0, 0.0, params, material)


def test_no_melt_pool_for_insufficient_power():
    # The 1/R singularity means any nonzero power produces melting arbitrarily
    # close to the source point, so "insufficient power" instead has to come from
    # an extreme velocity: the source moves away far faster than heat can
    # conduct outward, so the exponential decay term collapses even at the
    # near-source sampling point.
    material = get_material("AlSi10Mg")
    params = ProcessParameters(power=1.0, velocity=1e4, absorptivity=0.4)

    with pytest.raises(ValueError, match="No melt pool forms"):
        melt_pool_dimensions(params, material)


def test_unknown_material_raises():
    with pytest.raises(KeyError):
        get_material("Unobtanium")


def test_alpha_matches_definition():
    material = get_material("316L")
    assert material.alpha == pytest.approx(material.k / (material.rho * material.cp))
