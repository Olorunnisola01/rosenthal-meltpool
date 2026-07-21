"""Streamlit GUI for the rosenthal-meltpool package.

Deployed on Streamlit Community Cloud and embedded (via ?embed=true) on the
"Rosenthal Melt-Pool Calculator" project page of the portfolio site. Imports the
same rosenthal/ package used from the command line -- no duplicate physics here.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# rosenthal/ lives one directory up from this file, at the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rosenthal import MATERIALS, ProcessParameters, get_material, melt_pool_dimensions, temperature

st.set_page_config(page_title="Rosenthal Melt-Pool Calculator", layout="wide")

# Slider bounds, defined once so both the widgets and the fixed-window
# calculation below stay in sync.
POWER_RANGE = (50, 400)
VELOCITY_RANGE_MM_S = (100, 2000)
ABSORPTIVITY_RANGE = (0.1, 0.9)
T0_RANGE = (300, 800)


st.title("Rosenthal Melt-Pool Calculator")
st.caption(
    "Analytical (conduction-mode) prediction of single-track melt-pool geometry "
    "for laser powder bed fusion, from the Rosenthal (1946) moving point-heat-"
    "source solution."
)

with st.sidebar:
    st.header("Process parameters")
    material_name = st.selectbox("Material", list(MATERIALS.keys()))
    power = st.slider("Laser power (W)", min_value=POWER_RANGE[0], max_value=POWER_RANGE[1], value=200, step=5)
    velocity_mm_s = st.slider(
        "Scan velocity (mm/s)", min_value=VELOCITY_RANGE_MM_S[0], max_value=VELOCITY_RANGE_MM_S[1], value=800, step=10
    )
    absorptivity = st.slider(
        "Absorptivity", min_value=ABSORPTIVITY_RANGE[0], max_value=ABSORPTIVITY_RANGE[1], value=0.4, step=0.05
    )
    t0 = st.slider("Preheat temperature (K)", min_value=T0_RANGE[0], max_value=T0_RANGE[1], value=300, step=10)

material = get_material(material_name)
params = ProcessParameters(power=power, velocity=velocity_mm_s / 1000.0, absorptivity=absorptivity, t0=t0)

st.subheader(f"{material.name}")
st.caption(
    f"k = {material.k} W/m·K, ρ = {material.rho} kg/m³, Cp = {material.cp} J/kg·K, "
    f"T_melt = {material.t_melt} K — {material.source}"
)

try:
    dims = melt_pool_dimensions(params, material)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Width", f"{dims['width'] * 1e6:.1f} µm")
col2.metric("Depth", f"{dims['depth'] * 1e6:.1f} µm")
col3.metric("Length", f"{dims['length'] * 1e6:.1f} µm")
st.caption(
    "⚠️ Length is a known-weak output of the point-source Rosenthal model: the "
    "exponential decay term degenerates to 1 exactly on the trailing centerline, "
    "so length is systematically overpredicted. Treat width and depth as the "
    "reliable numbers; use length for relative comparisons only."
)

st.markdown("### Melt-pool geometry")

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    }
)

# The point-source singularity at R=0 makes raw temperature blow up to huge
# values right next to the source, which would crush the color scale near the
# melt pool into a single dark blob. Capping the display range at a modest
# multiple of the melt/ambient delta keeps contrast concentrated where it
# matters: in and around the melt-pool boundary.
delta = material.t_melt - t0
vmax_display = t0 + 2.2 * delta
levels = np.linspace(t0, vmax_display, 60)

# Axis limits (the "zoom") are recomputed from the current result every time,
# so the melt pool always fills the frame nicely at whatever size it actually
# is. What stays fixed is the figure's *rendered* size below (figsize=(6.5,
# 6.5), same physical plot area every time) -- only the numbers on the axes
# change, not the size of the plot itself.
half_w_fixed = dims["width"] / 2 * 1.6
x_back_fixed = dims["length_back"] * 0.35  # trailing tail is exaggerated (see caveat); crop it
x_front_fixed = dims["length_front"] * 1.8
depth_fixed = dims["depth"] * 1.8


def _field(coords_a, coords_b, plane: str) -> np.ndarray:
    """Evaluate temperature over a 2D grid, clipped for display."""
    A, B = np.meshgrid(coords_a, coords_b)
    if plane == "yz":  # cross-section through the source, x=0
        fn = lambda a, b: temperature(0.0, a, b, params, material)
    else:  # "xy", plan view at the surface, z=0
        fn = lambda a, b: temperature(a, b, 0.0, params, material)
    T = np.vectorize(lambda a, b: fn(a, b) if (a, b) != (0.0, 0.0) else np.nan)(A, B)
    return np.clip(T, t0, vmax_display)


col_plan, col_section = st.columns(2)

with col_plan:
    x = np.linspace(-x_back_fixed, x_front_fixed, 220)
    y = np.linspace(-half_w_fixed, half_w_fixed, 220)
    T_plan = _field(x, y, "xy")

    fig1, ax1 = plt.subplots(figsize=(6.5, 6.5))
    cf1 = ax1.contourf(x * 1e6, y * 1e6, T_plan, levels=levels, cmap="inferno", extend="max")
    ax1.contour(x * 1e6, y * 1e6, T_plan, levels=[material.t_melt], colors="#00e5ff", linewidths=2.5)
    ax1.plot(0, 0, marker="*", color="white", markersize=14, markeredgecolor="black", markeredgewidth=0.7)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x, scan direction (µm)")
    ax1.set_ylabel("y (µm)")
    ax1.set_title("Plan view (surface, z = 0)")
    st.pyplot(fig1, use_container_width=True)

with col_section:
    y2 = np.linspace(-half_w_fixed, half_w_fixed, 220)
    z2 = np.linspace(0, depth_fixed, 220)
    T_section = _field(y2, z2, "yz")

    fig2, ax2 = plt.subplots(figsize=(6.5, 6.5))
    cf2 = ax2.contourf(y2 * 1e6, z2 * 1e6, T_section, levels=levels, cmap="inferno", extend="max")
    ax2.contour(y2 * 1e6, z2 * 1e6, T_section, levels=[material.t_melt], colors="#00e5ff", linewidths=2.5)
    ax2.invert_yaxis()
    ax2.set_aspect("equal")
    ax2.set_xlabel("y (µm)")
    ax2.set_ylabel("depth z (µm)")
    ax2.set_title("Cross-section (x = 0)")
    st.pyplot(fig2, use_container_width=True)

fig_cb, ax_cb = plt.subplots(figsize=(10, 0.4))
fig_cb.subplots_adjust(bottom=0.6, top=0.95)
cb = fig_cb.colorbar(cf2, cax=ax_cb, orientation="horizontal")
cb.set_label(f"Temperature (K) — capped at display for contrast; cyan line = T_melt ({material.t_melt:.0f} K)")
st.pyplot(fig_cb, use_container_width=True)

st.caption(
    "The zoom level adapts to each result so the pool always fills the frame "
    "clearly; the plot's rendered size stays the same regardless. Plan-view "
    "trailing tail is cropped for display — the true isotherm runs far longer "
    "behind the source, per the length caveat above."
)

st.markdown("#### What these results mean")
st.markdown(
    f"""
- **Width ({dims['width'] * 1e6:.1f} µm)** sets the practical ceiling on **hatch
  spacing** — the spacing between adjacent laser passes. Hatch spacing is
  normally set to 50-80% of predicted width so adjacent tracks overlap enough to
  avoid unmelted gaps (lack-of-fusion porosity) between them.
- **Depth ({dims['depth'] * 1e6:.1f} µm)** must exceed the **powder layer
  thickness** (typically 20-60 µm in L-PBF) by a comfortable margin, so each new
  layer remelts into the one below it — this is what gives good interlayer
  bonding. If depth is only marginally larger than layer thickness, expect
  interlayer porosity.
- **Depth-to-width ratio is always exactly 0.5** — this is a mathematical
  identity of this model (see GitHub README), not a measurement: at the
  source's cross-section, the width and depth equations are literally the
  same equation, so the predicted pool is always a perfect semicircle. Real
  melt pools are not semicircular — conduction-mode pools run shallower than
  0.5, keyhole-mode pools run deeper — so this ratio cannot be used as a
  regime indicator here. Validating against [Zhang et al. (2024)](https://doi.org/10.3390/mi15020170)
  single-track 316L data shows measured ratios from 0.49 to 1.58 depending on
  parameters, none of which this model can reproduce by construction.
- **Length** is dominated by this model's known trailing-centerline artifact
  (see the caveat above) — it is not a reliable absolute number, so it isn't
  used for any of the interpretations above. It is included only for relative
  comparison between parameter sets, e.g. checking whether one parameter change
  elongates the pool more than another.
"""
)

st.markdown(
    """
    ---
    **Model limitations:** this is a conduction-mode analytical model — it does not
    capture keyhole-mode vaporization, recoil pressure, Marangoni convection, or
    temperature-dependent material properties. Treat results as first-pass
    estimates for comparing process parameters, not as a substitute for a
    validated thermal-fluid simulation. Absorptivity is the dominant tuning
    uncertainty; if predictions don't match an experimental single-track
    measurement, adjust absorptivity first.

    Source code: [GitHub](https://github.com/Olorunnisola01/rosenthal-meltpool) ·
    Reference: D. Rosenthal, *Trans. ASME*, 68, 849-866 (1946). Validated
    against [Zhang et al. (2024)](https://doi.org/10.3390/mi15020170)
    single-track 316L data — see the repo README for full results.
    """
)
