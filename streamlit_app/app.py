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

st.set_page_config(page_title="Rosenthal Melt-Pool Calculator", layout="centered")

st.title("Rosenthal Melt-Pool Calculator")
st.caption(
    "Analytical (conduction-mode) prediction of single-track melt-pool geometry "
    "for laser powder bed fusion, from the Rosenthal (1946) moving point-heat-"
    "source solution."
)

with st.sidebar:
    st.header("Process parameters")
    material_name = st.selectbox("Material", list(MATERIALS.keys()))
    power = st.slider("Laser power (W)", min_value=50, max_value=400, value=200, step=5)
    velocity_mm_s = st.slider("Scan velocity (mm/s)", min_value=100, max_value=2000, value=800, step=10)
    absorptivity = st.slider("Absorptivity", min_value=0.1, max_value=0.9, value=0.4, step=0.05)
    t0 = st.slider("Preheat temperature (K)", min_value=300, max_value=800, value=300, step=10)

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

st.markdown("### Melt-pool cross-section (through the source, x = 0)")

half_extent = max(dims["width"], dims["depth"]) * 1.5
n = 150
y = np.linspace(-half_extent, half_extent, n)
z = np.linspace(0, half_extent * 1.2, n)
Y, Z = np.meshgrid(y, z)
T = np.vectorize(lambda yy, zz: temperature(0.0, yy, zz, params, material) if (yy, zz) != (0.0, 0.0) else np.nan)(Y, Z)

fig, ax = plt.subplots(figsize=(5, 4))
contour = ax.contourf(Y * 1e6, Z * 1e6, T, levels=40, cmap="inferno")
ax.contour(Y * 1e6, Z * 1e6, T, levels=[material.t_melt], colors="cyan", linewidths=2)
ax.invert_yaxis()
ax.set_xlabel("y (µm)")
ax.set_ylabel("depth z (µm)")
ax.set_title("Temperature field (cyan = melt-pool boundary)")
fig.colorbar(contour, ax=ax, label="Temperature (K)")
st.pyplot(fig)

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

    Source code: [GitHub](https://github.com/) · Reference: D. Rosenthal,
    *Trans. ASME*, 68, 849-866 (1946).
    """
)
