"""Loader for Chen, Q., Zhao, Y., Strayer, S., Zhao, Y., Aoyagi, K., Koizumi,
Y., Chiba, A., Xiong, W., To, A.C. (2020), "Elucidating the Effect of
Preheating Temperature on Melt Pool Morphology Variation in Inconel 718
Laser Powder Bed Fusion via Simulation and Experiment," Additive
Manufacturing, doi: 10.1016/j.addma.2020.101642 (PII S2214860420310149).

Real (non-simulated) experimental single-track melt-pool measurements: EOS
M290 DMLS system, 400 W Yb fiber laser, 100 um focus diameter, IN718 bare
plate. 16 (power, velocity) combinations x 5 preheat temperatures
(100-500 C) = 80 ex-situ cross-sectional measurements total (Table A1,
Appendix), each giving both width and depth.

Only the 100 C column is used here (`load_chen2020_in718(preheat_c=100)`),
the lowest preheat tested and the closest match to the "near-ambient,
unheated substrate" convention of every other dataset in this project's
unified IN718/316L/Ti-6Al-4V/AlSi10Mg collection -- none of which record a
preheat temperature field, so mixing in the higher-preheat rows (200-500 C)
without extending SurrogateCase to carry that variable would silently
conflate a confounding factor this paper's own headline finding shows
matters a great deal (e.g. melt pool depth in keyhole mode increases up to
49% from 100 C to 500 C at fixed power/velocity). Loading the other 4
preheat columns is possible (`preheat_c` in {100, 200, 300, 400, 500}) for
a future, explicitly preheat-aware extension of this dataset.
"""

from dataclasses import dataclass

_VALID_PREHEATS = (100, 200, 300, 400, 500)

# (power_W, velocity_m_s, {preheat_C: (width_um, depth_um)})
_TABLE_A1 = [
    (200, 1.50, {100: (110.13, 50.22), 200: (111.89, 54.19), 300: (128.63, 58.59), 400: (128.67, 69.60), 500: (117.62, 65.64)}),
    (200, 1.00, {100: (107.72, 66.67), 200: (111.58, 75.44), 300: (115.09, 78.95), 400: (129.47, 86.67), 500: (130.87, 94.74)}),
    (200, 0.75, {100: (137.89, 106.67), 200: (138.47, 100.70), 300: (144.21, 107.02), 400: (167.02, 132.98), 500: (153.68, 135.44)}),
    (200, 0.50, {100: (165.96, 179.65), 200: (171.93, 172.98), 300: (188.77, 209.81), 400: (189.82, 209.81), 500: (177.19, 234.74)}),
    (250, 1.50, {100: (113.66, 59.74), 200: (122.91, 61.24), 300: (130.40, 74.01), 400: (129.96, 86.78), 500: (131.28, 88.99)}),
    (250, 1.00, {100: (114.16, 96.46), 200: (132.30, 101.76), 300: (136.28, 102.65), 400: (151.33, 125.66), 500: (150.44, 128.32)}),
    (250, 0.75, {100: (162.83, 131.42), 200: (156.63, 127.43), 300: (169.47, 150.00), 400: (165.93, 162.83), 500: (161.50, 176.11)}),
    (250, 0.50, {100: (186.73, 243.36), 200: (186.77, 236.28), 300: (201.77, 258.41), 400: (172.57, 295.58), 500: (176.99, 300.89)}),
    (285, 1.50, {100: (122.47, 73.57), 200: (126.43, 74.89), 300: (127.31, 84.14), 400: (125.55, 102.64), 500: (139.65, 95.59)}),
    (285, 1.00, {100: (129.20, 118.14), 200: (132.74, 111.50), 300: (140.26, 120.35), 400: (140.71, 136.28), 500: (153.98, 154.87)}),
    (285, 0.75, {100: (158.41, 163.27), 200: (165.93, 165.49), 300: (189.38, 182.74), 400: (180.53, 204.87), 500: (169.47, 200.00)}),
    (285, 0.50, {100: (161.94, 271.68), 200: (184.07, 259.29), 300: (202.65, 292.92), 400: (192.04, 340.71), 500: (191.92, 353.98)}),
    (350, 1.50, {100: (128.63, 98.24), 200: (134.80, 96.92), 300: (139.65, 113.66), 400: (131.28, 121.15), 500: (135.68, 130.40)}),
    (350, 1.00, {100: (150.89, 147.35), 200: (138.50, 134.51), 300: (153.10, 156.64), 400: (157.52, 180.53), 500: (166.37, 182.30)}),
    (350, 0.75, {100: (176.99, 200.89), 200: (180.53, 187.61), 300: (177.88, 209.73), 400: (172.57, 243.36), 500: (171.68, 252.21)}),
    (350, 0.50, {100: (161.95, 343.36), 200: (206.19, 340.71), 300: (199.12, 377.88), 400: (181.42, 407.96), 500: (184.96, 457.52)}),
]

_SPOT_DIAMETER_M = 100e-6


@dataclass(frozen=True)
class Chen2020Case:
    row_id: str
    power: float  # W
    velocity: float  # m/s
    beam_diameter: float  # m
    measured_width: float  # m
    measured_depth: float  # m
    preheat_c: int


def load_chen2020_in718(preheat_c: int = 100) -> list[Chen2020Case]:
    """Load Chen et al. (2020) IN718 single-track rows at one preheat
    temperature (default 100 C, the lowest tested; see module docstring).
    """
    if preheat_c not in _VALID_PREHEATS:
        raise ValueError(f"preheat_c must be one of {_VALID_PREHEATS}, got {preheat_c}")

    cases = []
    for i, (power, velocity, by_temp) in enumerate(_TABLE_A1):
        width_um, depth_um = by_temp[preheat_c]
        cases.append(
            Chen2020Case(
                row_id=f"chen2020_p{int(power)}_v{velocity}_t{preheat_c}",
                power=power,
                velocity=velocity,
                beam_diameter=_SPOT_DIAMETER_M,
                measured_width=width_um * 1e-6,
                measured_depth=depth_um * 1e-6,
                preheat_c=preheat_c,
            )
        )
    return cases
