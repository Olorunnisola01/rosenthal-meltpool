"""Compare model predictions against the NIST AM-Bench 2022 IN718 dataset.

Run: python scripts/validate_nist_ambench_IN718.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rosenthal import ProcessParameters, get_material, melt_pool_dimensions
from rosenthal.data.nist_ambench2022_in718 import NIST_AMBENCH_2022_IN718

ABSORPTIVITY = 0.4
T0 = 300.0


def main() -> None:
    material = get_material("IN718")
    print(f"{'Case':<6}{'P(W)':>7}{'v(m/s)':>9}{'Pi':>8}{'W_err%':>9}{'D_err%':>9}{'meas_AR':>9}")

    pis, w_errs = [], []
    for case in NIST_AMBENCH_2022_IN718:
        params = ProcessParameters(power=case.power, velocity=case.velocity, absorptivity=ABSORPTIVITY, t0=T0)
        pred = melt_pool_dimensions(params, material)
        pi = pred["width"] / case.spot_diameter
        w_err = (pred["width"] - case.measured_width) / case.measured_width * 100
        d_err = (pred["depth"] - case.measured_depth) / case.measured_depth * 100
        meas_ar = case.measured_depth / case.measured_width
        pis.append(pi)
        w_errs.append(w_err)
        print(f"{case.case_id:<6}{case.power:>7.0f}{case.velocity:>9.3f}{pi:>8.3f}{w_err:>9.1f}{d_err:>9.1f}{meas_ar:>9.2f}")

    r = np.corrcoef(pis, w_errs)[0, 1]
    print(f"\nPearson r (Pi vs width error): {r:.3f}")


if __name__ == "__main__":
    main()
