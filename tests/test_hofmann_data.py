import pytest

from rosenthal.data.hofmann2026_316L import load_hofmann_2026


def test_bare_plate_non_balling_count():
    cases = load_hofmann_2026()
    assert len(cases) == 172


def test_all_cases_are_bare_plate_and_not_balling():
    cases = load_hofmann_2026()
    for c in cases:
        assert c.layer_thickness == 0.0
        assert c.balling is False


def test_full_dataset_size_without_filters():
    cases = load_hofmann_2026(bare_plate_only=False, exclude_balling=False)
    assert len(cases) == 677


def test_spot_diameters_match_reported_levels():
    cases = load_hofmann_2026(bare_plate_only=False, exclude_balling=False)
    diameters_um = {round(c.spot_diameter * 1e6) for c in cases}
    assert diameters_um == {50, 80, 110, 140}


def test_case_fields_are_positive():
    cases = load_hofmann_2026()
    for c in cases[:20]:
        assert c.power > 0
        assert c.velocity > 0
        assert c.spot_diameter > 0
        assert c.measured_width > 0
        assert c.measured_depth > 0
