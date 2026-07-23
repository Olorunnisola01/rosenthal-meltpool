from rosenthal.data.nist_ambench2022_in718 import NIST_AMBENCH_2022_IN718


def test_seven_cases():
    assert len(NIST_AMBENCH_2022_IN718) == 7


def test_case_ids_match_paper_table():
    ids = {c.case_id for c in NIST_AMBENCH_2022_IN718}
    assert ids == {"0", "1.1", "1.2", "2.1", "2.2", "3.1", "3.2"}


def test_fields_positive_and_plausible_units():
    for c in NIST_AMBENCH_2022_IN718:
        assert 0 < c.power < 1000
        assert 0 < c.velocity < 10  # m/s
        assert 0 < c.spot_diameter < 1e-3  # metres, sub-mm spot
        assert 0 < c.measured_width < 1e-3
        assert 0 < c.measured_depth < 1e-3
