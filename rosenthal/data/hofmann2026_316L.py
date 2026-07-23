"""Loader for the Hofmann et al. (2026) 316L single-track dataset.

Source: Hofmann, M., Mayer, T., Friso, F., Radis, R., Kontermann, C., Muller, F.,
and Oechsner, M. (2026), "Melt pool geometry and process windows in PBF of 316L:
comprehensive single-source dataset and statistical modeling," Materials & Design,
262, 115459, doi: 10.1016/j.matdes.2026.115459. Dataset (CC-BY-4.0):
doi: 10.5281/zenodo.16979848, file MeltpoolGeometryData.csv.

677 single-track experiments on an Aconity-Midi machine, systematically varying
laser power, scan velocity, laser spot diameter (4 levels: 50/80/110/140 um), and
powder layer thickness (0/30/60 um), with measured weld width, penetration depth,
and a balling-instability flag.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

_CSV_PATH = Path(__file__).resolve().parent / "MeltpoolGeometryData.csv"


@dataclass(frozen=True)
class HofmannCase:
    """One single-track row from the Hofmann et al. (2026) dataset."""

    row_id: str
    power: float  # W
    velocity: float  # m/s
    spot_diameter: float  # m
    layer_thickness: float  # m
    measured_width: float  # m
    measured_depth: float  # m
    balling: bool


def load_hofmann_2026(bare_plate_only: bool = True, exclude_balling: bool = True) -> list[HofmannCase]:
    """Load the Hofmann et al. (2026) 316L dataset.

    Args:
        bare_plate_only: if True, keep only t_powder=0 rows (no powder layer),
            the cleanest comparison for a bare-plate point-source model.
        exclude_balling: if True, drop rows flagged as balling-unstable, a
            distinct failure mode (scan-track discontinuity) this model was
            never intended to capture.

    Returns:
        List of HofmannCase, sorted by row_id.
    """
    with open(_CSV_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()
    clean_lines = [line.replace('"', "") for line in lines]
    reader = csv.reader(clean_lines)
    rows = list(reader)
    header = rows[0]

    idx = {name: header.index(name) for name in header}

    cases = []
    for r in rows[1:]:
        layer_thickness = float(r[idx["t_powder [um]"]]) * 1e-6
        balling = r[idx["Balling"]] == "1"
        if bare_plate_only and layer_thickness != 0.0:
            continue
        if exclude_balling and balling:
            continue
        cases.append(
            HofmannCase(
                row_id=r[idx[""]],
                power=float(r[idx["P_laser [W]"]]),
                velocity=float(r[idx["v_scan [mm/s]"]]) / 1000.0,
                spot_diameter=float(r[idx["d_laser [mm]"]]) * 1e-3,
                layer_thickness=layer_thickness,
                measured_width=float(r[idx["Weldwidth w_w [um]"]]) * 1e-6,
                measured_depth=float(r[idx["Penetrationdepth d_w [um]"]]) * 1e-6,
                balling=balling,
            )
        )
    return cases
