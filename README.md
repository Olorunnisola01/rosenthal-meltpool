# rosenthal-meltpool

A plain Python implementation of the Rosenthal moving point-heat-source solution,
used to predict single-track melt-pool width, depth, and length for laser powder
bed fusion (L-PBF) process parameters.

Reference: D. Rosenthal, "The theory of moving sources of heat and its application
to metal treatments," *Trans. ASME*, 68, 849-866 (1946).

## What this is (and isn't)

This is a **conduction-mode** analytical model: it assumes the laser is a point
source moving at constant velocity over a semi-infinite solid, and that all
absorbed energy leaves the source purely by conduction. It has a closed-form
solution, which is what makes it fast enough for quick process-window screening,
but that closed form only exists for *constant* (temperature-independent)
material properties.

It does **not** model:
- Keyhole-mode vaporization or recoil pressure (deep, narrow pools at high power /
  low speed will be underestimated in depth).
- Marangoni convection within the melt pool.
- Temperature-dependent thermal conductivity, density, or specific heat.

Treat its output as a first-pass estimate for comparing process parameters against
each other, not as a substitute for a validated thermal-fluid simulation (e.g.
FVM/FEM with temperature-dependent properties) when precision matters.

**The predicted cross-section is always a perfect semicircle (depth/width ≡
0.5), for any material or process parameters.** This is not an approximation
— at the source's cross-section (x=0), the equations for width and for depth
are literally the same equation in the radial distance R, since
R=√(y²+z²) treats the width and depth directions identically. Real melt
pools are not semicircular (conduction-mode pools run shallower, keyhole-mode
pools run deeper), and this model cannot represent that variation at all —
see the validation section below for measured evidence of this gap.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ (uses `dict[str, float]` built-in generics) and `scipy`.

## Usage

```python
from rosenthal import ProcessParameters, get_material, melt_pool_dimensions

material = get_material("Ti-6Al-4V")
params = ProcessParameters(power=200.0, velocity=0.8, absorptivity=0.4, t0=300.0)

dims = melt_pool_dimensions(params, material)
print(dims)
# {'width': ..., 'depth': ..., 'length': ..., 'length_front': ..., 'length_back': ...}
# all dimensions in metres
```

`ProcessParameters`:

| Field | Meaning | Units |
|---|---|---|
| `power` | Nominal laser power | W |
| `velocity` | Scan velocity | m/s |
| `absorptivity` | Fraction of `power` actually absorbed (material/wavelength-dependent; typical L-PBF range 0.3-0.7) | - |
| `t0` | Preheat / build-plate temperature | K |

Preset materials (`rosenthal.MATERIALS`): `Ti-6Al-4V`, `316L`, `AlSi10Mg`, `IN718`.
Properties are temperature-averaged values from Mills (2002), *Recommended Values
of Thermophysical Properties for Selected Commercial Alloys* — see
`rosenthal/materials.py` for the exact numbers and per-alloy source note.

You can also evaluate the full temperature field directly:

```python
from rosenthal import temperature

T = temperature(x=-0.0002, y=0.0, z=0.0, params=params, material=material)
```

## `length` is a qualitative output, not a precise one

Exactly on the trailing centerline behind the source, the Rosenthal solution's
exponential decay term degenerates to 1 (see `melt_pool_dimensions` docstring for
why), so predicted temperature there falls off only as 1/R. This makes `length`
(and `length_back` specifically) substantially larger than real single-track
measurements, which use a distributed beam rather than an idealized point. Treat
`width` and `depth` as the trustworthy outputs of this model; use `length` only
for relative comparisons between parameter sets, not as an absolute prediction.

## Absorptivity is the dominant tuning uncertainty (but not a cure-all)

Everything else in this model is a material constant or a process parameter you
set directly. Absorptivity is the one input you cannot read off a datasheet — it
depends on powder morphology, layer thickness, wavelength, and angle of incidence.
If your predicted melt pool doesn't match an experimental single-track measurement,
absorptivity is the first parameter to check.

It is not, however, always enough. See the validation section below: for three of
four published test cases, no absorptivity value up to 500% closes the gap between
predicted and measured dimensions, because the point-source assumption itself
breaks down once predicted pool size approaches the real laser spot diameter.

## Project 2: parameter sweeps and literature validation

`rosenthal/sweep.py` evaluates `melt_pool_dimensions` over a power × velocity
grid (see `scripts/generate_process_maps.py`), producing process maps like the
ones in `figures/`. Because depth/width ≡ 0.5 always (see above), the aspect-ratio
panel of every process map is a flat 0.5 everywhere — that flatness is itself the
finding, not a rendering issue.

`rosenthal/validation.py` compares model predictions against a real published
dataset: **Zhang et al. (2024), "Understanding Melt Pool Behavior of 316L
Stainless Steel in Laser Powder Bed Fusion Additive Manufacturing," *Micromachines*
15(2):170**, [doi:10.3390/mi15020170](https://doi.org/10.3390/mi15020170) — four
bare-plate single-track cases (Table 3), using the paper's own assumed
absorptivity (0.5) and a 100 µm laser spot diameter.

Run `python scripts/validate_316L.py`:

| Case | P (W) | v (m/s) | Width pred. (µm) | Width meas. (µm) | Width err. | Depth pred. (µm) | Depth meas. (µm) | Depth err. | Meas. aspect ratio |
|---|---|---|---|---|---|---|---|---|---|
| N01 | 260 | 0.52 | 105.8 | 114.0 | -7.2% | 52.9 | 180.0 | -70.6% | 1.58 |
| N04 | 260 | 1.47 | 48.7 | 94.0 | -48.1% | 24.4 | 61.0 | -60.0% | 0.65 |
| N05 | 260 | 2.20 | 35.6 | 83.0 | -57.1% | 17.8 | 41.0 | -56.6% | 0.49 |
| N06 | 440 | 1.47 | 54.7 | 98.0 | -44.1% | 27.4 | 104.0 | -73.7% | 1.06 |

**Findings:**
- The model **underpredicts every case**, in both width and depth, at the
  paper's own assumed absorptivity.
- A calibration search (`calibrate_absorptivity`) shows this **cannot be fixed
  by absorptivity alone**: even searching up to 500% absorptivity, no solution
  exists that matches depth for *any* of the four cases, or width for three of
  the four. Only N01's width is reachable, at a calibrated 65.8% absorptivity.
- **Root cause:** predicted pool sizes here (35-115 µm) are the same order of
  magnitude as the paper's 100 µm laser spot diameter. The point-source
  assumption requires pool size ≫ source size to hold; these cases sit right at
  the boundary where that assumption is invalid.
- The measured aspect ratios (0.49 to 1.58) directly demonstrate the
  depth/width ≡ 0.5 limitation above: N01 and N06 are keyhole-influenced
  (aspect ratio > 1), a regime this conduction-mode point-source model cannot
  reach by construction, regardless of parameter tuning.

**What this means for using this model:** it is suitable for relative
comparisons between process parameters (screening "does A increase melt pool
size more than B") and for conduction-dominated regimes with pool sizes well
above the real beam spot size. It is not suitable for absolute dimensional
prediction validated against experiment, or for any parameter regime near or
in keyhole mode. A finite-source correction (e.g. a Gaussian surface flux
instead of a true point, or a Goldak double-ellipsoidal source) is the
natural next step to close this gap — noted here as a direction for future
work, not implemented in this package.

## Tests

```bash
pytest
```

## GUI

A Streamlit-based interactive GUI wrapping this same module lives in
`streamlit_app/app.py` and is deployed separately (see that folder's own notes).
This package has no GUI dependencies itself, by design — it stays a plain,
importable library.
