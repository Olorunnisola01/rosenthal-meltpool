"""Compare model predictions against Zhang et al. (2024) published 316L single-track data.

Run: python scripts/validate_316L.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rosenthal import ZHANG_2024_316L, calibrate_absorptivity, compare_to_case


def main() -> None:
    print("Direct comparison at the paper's own assumed absorptivity (0.5):\n")
    print(f"{'Case':<6}{'P(W)':>7}{'v(m/s)':>9}{'W_pred':>9}{'W_meas':>9}{'W_err%':>9}"
          f"{'D_pred':>9}{'D_meas':>9}{'D_err%':>9}{'AR_meas':>9}")
    for case in ZHANG_2024_316L:
        r = compare_to_case(case)
        print(
            f"{r['case_id']:<6}{r['power']:>7.0f}{r['velocity']:>9.2f}"
            f"{r['predicted_width'] * 1e6:>9.1f}{r['measured_width'] * 1e6:>9.1f}{r['width_error_pct']:>9.1f}"
            f"{r['predicted_depth'] * 1e6:>9.1f}{r['measured_depth'] * 1e6:>9.1f}{r['depth_error_pct']:>9.1f}"
            f"{r['measured_aspect_ratio']:>9.2f}"
        )

    print("\nCalibrated absorptivity search (bounds 0.01-5.0, i.e. up to 500%):\n")
    print(f"{'Case':<6}{'calib. for width':>18}{'calib. for depth':>18}")
    for case in ZHANG_2024_316L:
        a_w = calibrate_absorptivity(case, target="width")
        a_d = calibrate_absorptivity(case, target="depth")
        a_w_str = f"{a_w:.3f}" if a_w is not None else "no solution"
        a_d_str = f"{a_d:.3f}" if a_d is not None else "no solution"
        print(f"{case.case_id:<6}{a_w_str:>18}{a_d_str:>18}")

    print(
        "\n'No solution' means even a 500% absorptivity underpredicts that "
        "dimension -- absorptivity tuning cannot close the gap. This happens "
        "because the predicted pool sizes here (35-115 um) are comparable to "
        "the paper's 100 um laser spot diameter, violating the point-source "
        "model's basic validity assumption (pool size >> source size)."
    )


if __name__ == "__main__":
    main()
