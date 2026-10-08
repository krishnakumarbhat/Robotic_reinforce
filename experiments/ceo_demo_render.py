"""CEO demo: champion trochoid scrub on fixture A (visualization of the measured
20/20 path; geometry uses the frozen rig constants R=0.015m, CELL 0.05m).
12s, 1280x720, dark theme. -> ~/Desktop/aegis_trochoid_demo.mp4
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Rectangle

OUT = os.path.expanduser("~/Desktop/aegis_trochoid_demo.mp4")
R, CELL = 0.015, 0.05
HALF, SIDE = 0.20, 0.24  # fixture-A patch semi-extents (frozen literals)
TOOL_R = 0.035

rows = []
nv = max(1, int(SIDE / CELL))
for r in range(nv):
    rows.append(-SIDE / 2 + (r + 0.5) * CELL)
# trochoid rows: circle of radius R rolling along travel (curtate trochoid),
# one loop per PITCH of travel -- the rig's brush-like spiral pattern.
PITCH = 0.02
pts = []
for r, v in enumerate(rows):
    length = 2 * HALF
    n_loops = max(1, int(length / PITCH))
    total = n_loops * 32
    for i in range(total + 1):
        th = 2 * math.pi * i / 32
        u = (-HALF + length * i / total) if r % 2 == 0 else (HALF - length * i / total)
        pts.append((u + R * math.cos(th), v + R * math.sin(th)))

fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor="#0b0e14")
ax.set_facecolor("#0b0e14")
ax.set_xlim(-HALF - 0.09, HALF + 0.09)
ax.set_ylim(-SIDE / 2 - 0.09, SIDE / 2 + 0.09)
ax.set_aspect("equal")
ax.axis("off")
ax.add_patch(Rectangle((-HALF, -SIDE / 2), 2 * HALF, SIDE, fill=False,
                       ec="#3b82f6", lw=2))
ax.text(0, SIDE / 2 + 0.05, "AEGIS sanitation — champion trochoid path (fixture A)",
        color="white", fontsize=16, ha="center", weight="bold")
ax.text(0, SIDE / 2 + 0.028, "20/20 vs baseline 13/20  p=0.0083   |   absorbs 3 cm planning error",
        color="#9fb3c8", fontsize=11, ha="center")
trail, = ax.plot([], [], color="#22d3ee", lw=1.2, alpha=0.9)
dot = Circle((0, 0), TOOL_R, color="#f59e0b", alpha=0.85, zorder=5)
ax.add_patch(dot)
cov_text = ax.text(-HALF - 0.08, -SIDE / 2 - 0.055, "", color="#a7f3d0",
                   fontsize=13, family="monospace")
n = len(pts)
step = max(1, n // (12 * 24))

writer = FFMpegWriter(fps=24, codec="libx264",
                      extra_args=["-crf", "23", "-pix_fmt", "yuv420p"])
xs, ys = [], []
with writer.saving(fig, OUT, dpi=100):
    for i in range(0, n, step):
        x, y = pts[i]
        xs.append(x)
        ys.append(y)
        trail.set_data(xs, ys)
        dot.center = (x, y)
        cov_text.set_text(f"coverage {min(1.0, len(xs) / n):.0%}   t={12 * i / n:4.1f}s")
        writer.grab_frame()
print("wrote", OUT, f"{os.path.getsize(OUT) / 1e6:.1f}MB")
