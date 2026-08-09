"""Loader for Ti-6Al-4V and AlSi10Mg single-track PBF-LB/M rows drawn from the
MeltPoolNet aggregated melt-pool database (meltpoolnet_regression.csv).

meltpoolnet_regression.csv aggregates rows from many individual papers, tagged
by a "paper ID" column. We only keep rows whose paper ID resolves to a
verifiable, topic-matched primary source (checked against publisher/DOI
listings); rows tagged with a paper ID that had no resolvable URL in the
aggregated file (IDs 76 and 79) are dropped rather than cited blind. Exact
duplicate rows (same material/power/velocity/beamD/depth/width) are
deduplicated, keeping one instance.

Sources, by confidence tier (last checked 2026-08-06):

TIER 1 -- title, authors, journal, and DOI independently confirmed via
literature search (not just the aggregator's own PII/URL column):

Ti-6Al-4V (paper ID 2, 16 rows) -- EXPERIMENTAL (real measurement):
  - Dilip, J.J.S., Zhang, S., Teng, C., Zeng, K., Robinson, C., Pal, D., and
    Stucker, B. (2017), "Influence of processing parameters on the evolution
    of melt pool, porosity, and microstructures in Ti-6Al-4V alloy parts
    fabricated by selective laser melting," Progress in Additive
    Manufacturing, 2(3), 157-167, doi: 10.1007/s40964-017-0030-2.

Ti-6Al-4V (paper ID 15, 47 rows) -- upgraded from Tier 2 on 2026-08-06 after
direct inspection of the publisher PDF (user-supplied, institutional
access). IMPORTANT: this source is FE-SIMULATION output, not measurement --
discovered only on reading the full paper, since the aggregator does not
itself distinguish simulated from measured rows and this had previously
been assumed experimental (tagged accordingly in
`physics_informed._is_experimental_source`, now corrected):
  - Zhuang, J.-R., Lee, Y.-T., Hsieh, W.-H., Yang, A.-S. (2018),
    "Determination of melt pool dimensions using DOE-FEM and RSM with
    process window during SLM of Ti6Al4V powder," Optics and Laser
    Technology, 103, 59-76, doi: 10.1016/j.optlastec.2018.01.013. A
    transient ANSYS finite-element thermal model, fit via a 4-factor,
    7-level central composite design-of-experiments (49 design points --
    matches this loader's 47-row count after deduplication) to predict
    melt-pool length/width/depth as a function of laser power, scan speed,
    preheat temperature, and hatch spacing. The FEM was itself validated
    against separately published experimental/computational results (not
    reproduced here), but the rows in meltpoolnet_regression.csv for this
    paper are the FEM's own simulated output, not physical measurements.
    NOTE: journal name correction -- this is "Optics and Laser Technology"
    (ISSN 0030-3992), NOT "Optics and Lasers in Engineering" (ISSN
    0143-8166, a different journal); an earlier round of this project's
    citation work conflated the two similarly-named journals when
    searching for this citation blind.

  Consequence for Ti-6Al-4V's dataset composition: only 16 of 63 (25%) of
  this project's "Ti-6Al-4V" cases are real measurements; 47 (75%) are FE
  simulation, comparable in severity to the IN718 sim/real imbalance found
  in round 2 (which was subsequently addressed there but not, until this
  discovery, checked for Ti-6Al-4V). A sim-to-real transfer test analogous
  to IN718's (train on the 47 simulated cases, test on the 16 real ones)
  shows the same catastrophic failure pattern: width R^2 = -4.06, depth
  R^2 = -0.14. This is very likely a significant, previously-unaccounted-
  for contributor to Ti-6Al-4V's consistently poor performance throughout
  this project (rounds 2, 5, 8, 9 all found Ti-6Al-4V the worst-performing
  alloy and attributed it primarily to its distinct absorptivity regime;
  this data-composition problem is now a second, likely compounding,
  explanation that should be reported alongside it).

AlSi10Mg (paper ID 3, 6 rows):
  - Yu, W. et al. (2016), "Influence of processing parameters on laser
    penetration depth and melting/re-melting densification during selective
    laser melting of aluminum alloy," Applied Physics A, 122, doi:
    10.1007/s00339-016-0428-6.

AlSi10Mg (paper ID 12, 14 rows) -- upgraded from Tier 2 on 2026-08-06 after
direct inspection of the publisher PDF (user-supplied, institutional
access):
  - Guo, Q., Zhao, C., Qu, M., Xiong, L., Escano, L.I., Hojjatzadeh,
    S.M.H., Parab, N.D., Fezzaa, K., Everhart, W., Sun, T., Chen, L.
    (2019), "In-situ characterization and quantification of melt pool
    variation under constant input energy density in laser powder bed
    fusion additive manufacturing process," Additive Manufacturing, 28,
    600-609, doi: 10.1016/j.addma.2019.04.021. Confirmed on-topic and
    parameter-range-matched: AlSi10Mg powder bed, laser powers 104-520 W,
    D4-sigma beam diameter 100 um (matches this loader's assumed beam
    diameter for these rows), single-track melt-pool geometry measured by
    high-speed synchrotron x-ray imaging.

(Historical note: paper ID 15 was Tier 2 through several rounds of this
project. Eight non-paywalled verification methods failed to resolve its
title; a first user-supplied PDF for a plausible candidate PII
(S0143816617313246) turned out to be an unrelated paper -- Zhang, S.
(2018), "High-speed 3D shape measurement with structured light methods: A
review," Optics and Lasers in Engineering, 106, 119-131 (a DIFFERENT
journal from the one paper ID 15 is actually published in -- see the
journal-name correction above), a 3D-imaging review with no melt-pool
content. The correct PII, S0030399217306400, was
resolved via a second user-supplied PDF on 2026-08-06 and is now the
Tier-1 Zhuang et al. citation above.)

IN718 -- real (non-simulated) experimental data, added specifically to
reduce the FE-simulation share of the IN718 dataset (see the sim-to-real
diagnostic in scripts/train_physics_informed.py):

TIER 1:
IN718 (paper ID 9, 5 rows -- all with beam diameter reported):
  - "Melt pool geometry and morphology variability for the Inconel 718
    alloy in a laser powder bed fusion additive manufacturing process,"
    Additive Manufacturing, 2019, pii S2214860419306104. Directly on-topic
    (title independently confirmed by search; full author/volume/page
    details not extracted -- resolve before final manuscript citation).

TIER 2 (real, indexed article confirmed via publisher/aggregator redirect;
exact title not independently confirmed):
IN718 (paper ID 37, 13 rows -- all with beam diameter reported):
  - Indexed via Gale OneFile Academic (ID A648617896); ISSN 18800688
    (Optics & Laser Technology). Do not upgrade to a Tier-1 citation in the
    manuscript without manual verification.

(Paper IDs 11 and 33 also tag real IN718 rows in this file, but every one of
those rows is missing beam diameter -- a required feature for this model --
so they were not usable regardless of citation tier.)

Rows tagged with a paper ID that had no resolvable URL at all in the
aggregated file (IDs 76 and 79) are dropped rather than cited blind. Exact
duplicate rows (same material/power/velocity/beamD/depth/width) are
deduplicated, keeping one instance. Only PBF/SLM (laser powder bed fusion)
rows are kept -- DED/EBM rows in the same material are excluded since they
are a different process with different absorptivity/heat-source physics.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

_CSV_PATH = Path(__file__).resolve().parent / "meltpoolnet_regression.csv"

_VERIFIED_PAPER_IDS = {
    "Ti-6Al-4V": {"2", "15"},
    "AlSi10Mg": {"3", "12"},
    "IN718": {"9", "37"},
}


@dataclass(frozen=True)
class MeltPoolNetCase:
    """One single-track PBF/SLM row from the MeltPoolNet aggregated dataset."""

    row_id: str
    alloy: str
    power: float  # W
    velocity: float  # m/s
    spot_diameter: float  # m
    measured_width: float  # m
    measured_depth: float  # m
    paper_id: str


def load_meltpoolnet_alloy(alloy: str) -> list[MeltPoolNetCase]:
    """Load verified PBF/SLM single-track rows for one alloy.

    Args:
        alloy: "Ti-6Al-4V" or "AlSi10Mg".

    Returns:
        List of MeltPoolNetCase, deduplicated, sorted by row index.
    """
    if alloy not in _VERIFIED_PAPER_IDS:
        raise ValueError(f"No verified paper IDs configured for alloy {alloy!r}")

    with open(_CSV_PATH, encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = rows[0]
    idx = {name: header.index(name) for name in header}

    seen = set()
    cases = []
    for i, r in enumerate(rows[1:]):
        if r[idx["Material"]] != alloy:
            continue
        if r[idx["Process"]] != "PBF" or r[idx["Sub-process"]] != "SLM":
            continue
        paper_id = r[idx["paper ID"]].strip()
        if paper_id not in _VERIFIED_PAPER_IDS[alloy]:
            continue
        depth_raw = r[idx["depth of meltpool"]].strip()
        width_raw = r[idx["width of melt pool"]].strip()
        beam_raw = r[idx["beam D"]].strip()
        power_raw = r[idx["Power"]].strip()
        velocity_raw = r[idx["Velocity"]].strip()
        if not (depth_raw and width_raw and beam_raw and power_raw and velocity_raw):
            continue

        dedup_key = (alloy, power_raw, velocity_raw, beam_raw, depth_raw, width_raw)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        cases.append(
            MeltPoolNetCase(
                row_id=f"meltpoolnet_{alloy}_{i}",
                alloy=alloy,
                power=float(power_raw),
                velocity=float(velocity_raw) * 1e-3,
                spot_diameter=float(beam_raw) * 1e-6,
                measured_width=float(width_raw) * 1e-6,
                measured_depth=float(depth_raw) * 1e-6,
                paper_id=paper_id,
            )
        )
    return cases
