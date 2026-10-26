import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
"""README charts (matplotlib): search progress, lift vs where the box ends up, geometry sketch."""
import sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as pa
OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/images"; os.makedirs(OUT, exist_ok=True)
BG, FG, GR = "#0f141a", "#e6e9ee", "#2b3440"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GR, "axes.labelcolor": FG, "text.color": FG,
                     "xtick.color": FG, "ytick.color": FG, "grid.color": GR, "font.size": 10})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
# 1  CMA progress
g = ["start"] + [str(i) for i in range(7)]; sc = [6.0, 13.1, 15.7, 17.4, 19.9, 19.9, 19.9, 19.9]
ax[0].plot(range(len(sc)), sc, "-o", color="#ff9d2e"); ax[0].set_xticks(range(len(sc))); ax[0].set_xticklabels(g)
ax[0].set_xlabel("search generation"); ax[0].set_ylabel("score (higher is better)"); ax[0].set_title("Searching the wrap settings"); ax[0].grid(True, alpha=.4)
# 2  lift vs where the box is when the trunk starts to lower
x = [0.381, 0.395, 0.427, 0.438, 0.450, 0.461]
base = [-8.31, 3.68, 7.07, 6.59, -8.31, -8.31]; tuned = [-0.06, 0.1, 9.62, 9.23, 9.35, 10.42]
ax[1].plot(x, base, "s--", color="#8aa0b8", label="hand-tuned settings (before)"); ax[1].plot(x, tuned, "o-", color="#ff9d2e", label="searched settings (after)")
ax[1].axhline(0, color=GR); ax[1].set_xlabel("box distance ahead of torso when the trunk starts to lower (m)"); ax[1].set_ylabel("final box height gain (cm)\n(-8 = knocked over)")
ax[1].set_title("Same six saved starts, two settings"); ax[1].legend(facecolor=BG, edgecolor=GR); ax[1].grid(True, alpha=.4)
# 3  full pipeline runs
runs = ["nominal", "+3 cm x\n+2 cm y", "-2 cm x\n-2 cm y", "heading\n+8 deg", "heading -8\n+3 cm y", "+10 cm x\n-5 cm y"]; lift = [9.40, 9.47, 9.68, 9.61, 9.13, 9.73]
ax[2].bar(range(6), lift, color="#ff9d2e"); ax[2].set_xticks(range(6)); ax[2].set_xticklabels(runs, fontsize=8); ax[2].set_ylim(0, 12)
ax[2].set_ylabel("box height gain (cm)"); ax[2].set_title("Full camera pipeline, 6 of 6 runs lifted"); ax[2].grid(True, axis="y", alpha=.4)
for i, v in enumerate(lift): ax[2].text(i, v + .2, f"{v:.1f}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/results.png", dpi=130); plt.close()

# geometry sketch: top view, torso frame (x forward = up the page)
fig, a = plt.subplots(figsize=(6.4, 6.4)); a.set_aspect("equal"); a.set_xlim(0.45, -0.45); a.set_ylim(-0.35, 1.0)
a.add_patch(pa.Rectangle((-0.12, -0.30), 0.24, 0.70, color="#5d5648", label="body and head (schematic)"))
a.plot([0, 0], [0.40, 0.89], color="#ff7a1a", lw=7, solid_capstyle="round", label="trunk, root at the head, tip 0.89 m ahead")
a.add_patch(pa.Rectangle((0.19 - 0.0175, 0.52 - 0.0175), 0.035, 0.035, color="#6ea8ff"))
a.add_patch(pa.Rectangle((0.075 - 0.0175, 0.45 - 0.0175), 0.035, 0.035, color="#7dff8a"))
a.text(0.44, 0.80, "1  park: box 0.52 m ahead,\n    about 20 deg to the left", color="#6ea8ff", fontsize=9)
a.text(0.44, 0.68, "2  after the turn: box 0.45 m ahead,\n    0.075 m to the left of the trunk", color="#7dff8a", fontsize=9)
a.annotate("", xy=(0.085, 0.475), xytext=(0.18, 0.525), arrowprops=dict(arrowstyle="->", color=FG)); a.text(0.44, 0.57, "the robot spins in place;\nthat moves the box toward the trunk", color=FG, fontsize=9)
a.set_xlabel("sideways (m), left is positive"); a.set_ylabel("forward (m)"); a.set_title("Top view: where the box sits relative to the robot"); a.grid(True, alpha=.4); a.legend(loc="lower right", facecolor=BG, edgecolor=GR, fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/geometry.png", dpi=130)
print("charts done")
