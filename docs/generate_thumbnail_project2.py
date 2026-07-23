"""Generate a portfolio thumbnail for the Project 2 page (process maps +
literature validation), matching the flowchart/pastel-box style used across
the site's other project thumbnails and Project 1's own thumbnail.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rosenthal import ZHANG_2024_316L, compare_to_case, get_material, sweep

OUT_PATH = Path(__file__).resolve().parent.parent.parent / "portfolio-website-src" / "content" / "project" / "Meltpool_Process_Validation" / "featured.png"

fig, ax = plt.subplots(figsize=(14, 7.0), dpi=110)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.0)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.add_patch(Rectangle((0.05, 0.05), 13.9, 6.9, fill=False, edgecolor="black", linewidth=2.5))

title_box = FancyBboxPatch(
    (2.2, 6.05), 9.6, 0.7,
    boxstyle="round,pad=0.05,rounding_size=0.08",
    facecolor="#f2f2f2", edgecolor="black", linewidth=1.5,
)
ax.add_patch(title_box)
ax.text(7.0, 6.4, "PROCESS MAPS & LITERATURE VALIDATION", ha="center", va="center",
         fontsize=16.5, fontweight="bold", color="#7a1f1f", family="serif")

# --- SWEEP box (cream) ---
sweep_box = FancyBboxPatch(
    (0.5, 1.3), 3.9, 4.4,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    facecolor="#fdeec6", edgecolor="#c9a227", linewidth=1.8,
)
ax.add_patch(sweep_box)
ax.text(2.45, 5.3, "SWEEP", ha="center", va="center", fontsize=15, fontweight="bold")
ax.text(2.45, 4.85, "Power x Velocity Grid", ha="center", va="center", fontsize=10.5, style="italic")

material = get_material("316L")
powers = np.linspace(50, 400, 40)
velocities = np.linspace(0.2, 2.0, 40)
result = sweep(material, powers, velocities, absorptivity=0.4, t0=300.0)
icon_ax = ax.inset_axes([0.9, 2.7, 3.0, 1.9], transform=ax.transData)
icon_ax.pcolormesh(powers, velocities * 1000, result["width"] * 1e6, cmap="viridis", shading="auto")
icon_ax.set_xticks([])
icon_ax.set_yticks([])
for spine in icon_ax.spines.values():
    spine.set_edgecolor("#7a5c00")
    spine.set_linewidth(1.2)

ax.text(2.45, 2.3, "4 alloys x width,\ndepth, aspect ratio", ha="center", va="center", fontsize=10.5)

# --- VALIDATE box (blue) ---
val_box = FancyBboxPatch(
    (4.75, 1.3), 4.5, 4.4,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    facecolor="#eaf2fb", edgecolor="#2b6cb0", linewidth=1.8,
)
ax.add_patch(val_box)
ax.text(7.0, 5.3, "VALIDATE", ha="center", va="center", fontsize=15, fontweight="bold")
ax.text(7.0, 4.85, "vs. Zhang et al. (2024)", ha="center", va="center", fontsize=10.5, style="italic")

# Left edge pulled in further from the box border (was 5.1, only 0.35 units
# from the box's 4.75 edge -- too tight for a rotated ylabel + tick labels,
# which made "Width (um)" look jammed against the border). Width reduced to
# compensate so the chart's right edge still sits comfortably inside the box.
bar_ax = ax.inset_axes([5.55, 2.05, 3.45, 2.35], transform=ax.transData)
cases = ZHANG_2024_316L
labels = [c.case_id for c in cases]
pred = [compare_to_case(c)["predicted_width"] * 1e6 for c in cases]
meas = [compare_to_case(c)["measured_width"] * 1e6 for c in cases]
x = np.arange(len(cases))
bar_ax.bar(x - 0.18, pred, width=0.36, color="#2b6cb0", label="Predicted")
bar_ax.bar(x + 0.18, meas, width=0.36, color="#e07b39", label="Measured")
bar_ax.set_xticks(x)
bar_ax.set_xticklabels(labels, fontsize=8)
bar_ax.set_ylabel("Width (µm)", fontsize=7.5, labelpad=2)
bar_ax.tick_params(axis="y", labelsize=6.5)
bar_ax.legend(fontsize=6.5, loc="upper right", frameon=False)
for spine in ["top", "right"]:
    bar_ax.spines[spine].set_visible(False)

ax.text(7.0, 1.55, "Model underpredicts width in every case", ha="center", va="center", fontsize=9.5, color="#b3401f")

# --- FINDING box (green) ---
find_box = FancyBboxPatch(
    (9.55, 1.3), 3.9, 4.4,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    facecolor="#e2f3e2", edgecolor="#3a8f3a", linewidth=1.8,
)
ax.add_patch(find_box)
ax.text(11.5, 5.3, "KEY FINDINGS", ha="center", va="center", fontsize=14, fontweight="bold")
findings = [
    "Depth/width ≡ 0.5\nacross the entire\nprocess window",
    "",
    "Absorptivity search\nup to 500%: no\nsolution for 3/4\ncases",
    "",
    "Root cause: pool size\n≈ 100 µm laser spot",
]
y0 = 4.6
for line in findings:
    ax.text(11.5, y0, line, ha="center", va="center", fontsize=10)
    y0 -= 0.62 if line else 0.35

# Arrows
arrow_style = dict(arrowstyle="-|>", color="black", linewidth=2, mutation_scale=22)
ax.add_patch(FancyArrowPatch((4.45, 3.5), (4.7, 3.5), **arrow_style))
ax.add_patch(FancyArrowPatch((9.3, 3.5), (9.5, 3.5), **arrow_style))

tag = FancyBboxPatch((10.5, 0.4), 3.4, 0.6, boxstyle="square,pad=0.02", facecolor="#595959", edgecolor="none")
ax.add_patch(tag)
ax.text(12.2, 0.7, "Process Validation", ha="center", va="center", fontsize=11, color="white", fontweight="bold")

fig.tight_layout(pad=0)
fig.savefig(OUT_PATH, dpi=110, facecolor="white")
print(f"Saved {OUT_PATH}")
