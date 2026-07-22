"""Generate a portfolio thumbnail for the Rosenthal calculator project page,
matching the flowchart/diagram style used by the site's other project
thumbnails (rounded pastel boxes, arrows, boxed title, small bottom-right tag).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rosenthal import ProcessParameters, get_material, melt_pool_dimensions, temperature

OUT_PATH = Path(__file__).resolve().parent.parent.parent / "portfolio-website-src" / "content" / "project" / "Rosenthal_Meltpool_Calculator" / "featured.png"

fig, ax = plt.subplots(figsize=(14, 7.0), dpi=110)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.0)
ax.axis("off")
fig.patch.set_facecolor("white")

# Outer border (matches the thin black frame seen on other thumbnails)
ax.add_patch(Rectangle((0.05, 0.05), 13.9, 6.9, fill=False, edgecolor="black", linewidth=2.5))

# Title box
title_box = FancyBboxPatch(
    (2.7, 6.05), 8.6, 0.7,
    boxstyle="round,pad=0.05,rounding_size=0.08",
    facecolor="#f2f2f2", edgecolor="black", linewidth=1.5,
)
ax.add_patch(title_box)
ax.text(7.0, 6.4, "ANALYTICAL MELT-POOL PREDICTION", ha="center", va="center",
         fontsize=17, fontweight="bold", color="#7a1f1f", family="serif")

# --- INPUT box (cream/yellow, like the GA/optimization boxes) ---
input_box = FancyBboxPatch(
    (0.5, 1.3), 3.6, 4.4,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    facecolor="#fdeec6", edgecolor="#c9a227", linewidth=1.8,
)
ax.add_patch(input_box)
ax.text(2.3, 5.3, "INPUT", ha="center", va="center", fontsize=15, fontweight="bold")
ax.text(2.3, 4.8, "Process Parameters", ha="center", va="center", fontsize=11.5, style="italic")
lines = ["Power  P  (W)", "Velocity  v  (m/s)", "Absorptivity  η", "Preheat  T₀  (K)", "Material", "(k, ρ, cp, T_melt)"]
for i, line in enumerate(lines):
    ax.text(2.3, 4.2 - i * 0.5, line, ha="center", va="center", fontsize=11)

# --- MODEL box (center, white/blue with the equation) ---
model_box = FancyBboxPatch(
    (4.6, 1.3), 4.8, 4.4,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    facecolor="#eaf2fb", edgecolor="#2b6cb0", linewidth=1.8,
)
ax.add_patch(model_box)
ax.text(7.0, 5.3, "ROSENTHAL (1946)", ha="center", va="center", fontsize=14.5, fontweight="bold")
ax.text(7.0, 4.8, "Moving Point-Heat-Source Solution", ha="center", va="center", fontsize=10.5, style="italic")
ax.text(
    7.0, 3.8,
    r"$T - T_0 = \dfrac{Q}{2\pi k R}\,\exp\!\left(-\dfrac{v(R+x)}{2\alpha}\right)$",
    ha="center", va="center", fontsize=15,
)
ax.text(7.0, 2.8, r"$R=\sqrt{x^2+y^2+z^2}$" + "   " + r"$\alpha = k/(\rho c_p)$", ha="center", va="center", fontsize=12)
ax.text(7.0, 2.0, "Solved numerically for the\n" + r"$T = T_{melt}$" + " isotherm", ha="center", va="center", fontsize=10.5, color="#444444")

# --- OUTPUT box (green, like the "BEST" result box) ---
output_box = FancyBboxPatch(
    (9.9, 1.3), 3.6, 4.4,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    facecolor="#e2f3e2", edgecolor="#3a8f3a", linewidth=1.8,
)
ax.add_patch(output_box)
ax.text(11.7, 5.3, "OUTPUT", ha="center", va="center", fontsize=15, fontweight="bold")
ax.text(11.7, 4.8, "Melt-Pool Geometry", ha="center", va="center", fontsize=11.5, style="italic")

# Small semicircle melt-pool icon echoing the calculator's own plot (cyan boundary + star)
material = get_material("AlSi10Mg")
params = ProcessParameters(power=200.0, velocity=0.8, absorptivity=0.4, t0=300.0)
dims = melt_pool_dimensions(params, material)
icon_cx, icon_cy = 11.7, 3.35
scale = 1.15 / (dims["width"] / 2)
theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
half_w = dims["width"] / 2 * scale
depth = dims["depth"] * scale
xs = icon_cx + depth * np.cos(theta) * (-1)
ys = icon_cy + half_w * np.sin(theta)
ax.fill(np.concatenate([xs, [icon_cx]]), np.concatenate([ys, [icon_cy]]), color="#ffd76b", alpha=0.9, zorder=3)
ax.plot(xs, ys, color="#00b3c6", linewidth=2.4, zorder=4)
ax.plot([icon_cx], [icon_cy], marker="*", color="white", markersize=11, markeredgecolor="black", markeredgewidth=0.6, zorder=5)

ax.text(11.7, 1.9, "Width · Depth · Length", ha="center", va="center", fontsize=11)

# Arrows
arrow_style = dict(arrowstyle="-|>", color="black", linewidth=2, mutation_scale=22)
ax.add_patch(FancyArrowPatch((4.15, 3.5), (4.55, 3.5), **arrow_style))
ax.add_patch(FancyArrowPatch((9.45, 3.35), (9.85, 3.35), **arrow_style))

# Bottom-right tag, matching the small gray label seen on other thumbnails -- sits
# entirely in the margin below the content boxes, no overlap.
tag = FancyBboxPatch((10.55, 0.4), 3.4, 0.6, boxstyle="square,pad=0.02", facecolor="#595959", edgecolor="none")
ax.add_patch(tag)
ax.text(12.25, 0.7, "Melt-Pool Model", ha="center", va="center", fontsize=11.5, color="white", fontweight="bold")

fig.tight_layout(pad=0)
fig.savefig(OUT_PATH, dpi=110, facecolor="white")
print(f"Saved {OUT_PATH}")
