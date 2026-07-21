import numpy as np
import pytest

from rosenthal import get_material, sweep


def test_sweep_shape_and_monotonic_trend():
    material = get_material("Ti-6Al-4V")
    powers = np.array([100.0, 200.0, 300.0])
    velocities = np.array([0.4, 0.8, 1.2])

    result = sweep(material, powers, velocities, absorptivity=0.4, t0=300.0)

    for key in ("width", "depth", "length", "aspect_ratio"):
        assert result[key].shape == (3, 3)

    # Width should increase with power at fixed velocity (row 0 = lowest velocity).
    assert result["width"][0, 0] < result["width"][0, 1] < result["width"][0, 2]

    # Width should decrease with velocity at fixed power (column 0 = lowest power).
    assert result["width"][0, 0] > result["width"][1, 0] > result["width"][2, 0]


def test_sweep_nan_for_sub_threshold_parameters():
    material = get_material("AlSi10Mg")
    powers = np.array([1.0])
    velocities = np.array([1e4])  # deliberately extreme, see test_rosenthal.py

    result = sweep(material, powers, velocities, absorptivity=0.4, t0=300.0)

    assert np.isnan(result["width"][0, 0])
    assert np.isnan(result["aspect_ratio"][0, 0])
