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

## Absorptivity is the dominant tuning uncertainty

Everything else in this model is a material constant or a process parameter you
set directly. Absorptivity is the one input you cannot read off a datasheet — it
depends on powder morphology, layer thickness, wavelength, and angle of incidence.
If your predicted melt pool doesn't match an experimental single-track measurement,
absorptivity is the first parameter to adjust, not the material properties.

## Tests

```bash
pytest
```

## GUI

A Streamlit-based interactive GUI wrapping this same module lives in
`streamlit_app/app.py` and is deployed separately (see that folder's own notes).
This package has no GUI dependencies itself, by design — it stays a plain,
importable library.
