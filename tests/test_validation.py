import pytest

from rosenthal import ZHANG_2024_316L, calibrate_absorptivity, compare_to_case


def test_zhang_2024_dataset_has_four_cases():
    assert len(ZHANG_2024_316L) == 4
    assert {c.case_id for c in ZHANG_2024_316L} == {"N01", "N04", "N05", "N06"}


def test_compare_to_case_returns_expected_keys():
    case = ZHANG_2024_316L[0]
    result = compare_to_case(case)
    for key in ("predicted_width", "measured_width", "width_error_pct", "predicted_depth", "measured_depth", "depth_error_pct"):
        assert key in result


def test_model_underpredicts_all_zhang_cases_at_paper_absorptivity():
    # This is the actual finding, not an assumption: at the paper's own
    # absorptivity (0.5), the point-source model underpredicts both width and
    # depth for every published case, because predicted pool sizes are on the
    # same order as the paper's 100 um laser spot -- see calibration test below.
    for case in ZHANG_2024_316L:
        result = compare_to_case(case)
        assert result["width_error_pct"] < 0
        assert result["depth_error_pct"] < 0


def test_calibration_cannot_close_gap_for_most_cases():
    # Confirms that the underprediction is not just a wrong absorptivity
    # guess: even searching up to 500% absorptivity, depth cannot be matched
    # for any of the four cases, and width cannot be matched for three of
    # four (N01, the keyhole-influenced case, is the sole exception).
    depth_solutions = [calibrate_absorptivity(c, target="depth") for c in ZHANG_2024_316L]
    assert all(sol is None for sol in depth_solutions)

    width_solutions = {c.case_id: calibrate_absorptivity(c, target="width") for c in ZHANG_2024_316L}
    assert width_solutions["N01"] is not None
    assert width_solutions["N04"] is None
    assert width_solutions["N05"] is None
    assert width_solutions["N06"] is None
