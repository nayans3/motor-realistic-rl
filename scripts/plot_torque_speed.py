"""Plot the Go1 joint torque-speed curves: the 'ideal' simulator vs the motor model.

Run from the repo root:  python scripts/plot_torque_speed.py
Writes results/torque_speed_curve.png
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from motor_model import DEFAULT_PARAMS, torque_speed_curve  # noqa: E402

BLUE, ORANGE = "#257de0", "#f47230"          # categorical slots 1 and 2
GRAY, INK, INK2 = "#9c9c9c", "#191919", "#666666"

P = DEFAULT_PARAMS
speeds = np.linspace(0, 40, 401)

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(GRAY)
ax.tick_params(colors=INK2, labelsize=9)
ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
ax.set_axisbelow(True)

joints = [("Hip / thigh", False, BLUE), ("Knee", True, ORANGE)]
for name, knee, color in joints:
    cap = P.tau_max * (1.5 if knee else 1.0)
    full = np.asarray(torque_speed_curve(speeds, params=P, knee=knee))
    low = np.asarray(torque_speed_curve(speeds, v_bus=18.0, params=P, knee=knee))
    # Ideal simulator: a flat cap at every speed.
    ax.plot(speeds, np.full_like(speeds, cap), color=GRAY, lw=1.2, ls=(0, (2, 2)))
    ax.plot(speeds, full, color=color, lw=2, label=f"{name}, 24 V battery")
    ax.plot(speeds, low, color=color, lw=2, ls=(0, (5, 3)), label=f"{name}, 18 V (low battery)")
    ax.text(40, cap + 0.6, f"Ideal sim: {cap:g} N·m at any speed", color=INK2,
            fontsize=8.5, ha="right", va="bottom")
    # Direct label at the knee of the full-battery curve.
    i = np.argmax(full < cap - 0.5)
    ax.annotate(name, (speeds[i], full[i]), xytext=(8, 4), textcoords="offset points",
                color=INK, fontsize=9, fontweight="bold")

ax.set_xlim(0, 40)
ax.set_ylim(0, 40)
ax.set_xlabel("Joint speed (rad/s)", color=INK2, fontsize=9)
ax.set_ylabel("Max torque available (N·m)", color=INK2, fontsize=9)
ax.set_title("Real motors lose torque at speed; the ideal simulator never does",
             color=INK, fontsize=12, fontweight="bold", loc="left", pad=22)
ax.text(0, 1.02, "Go1 joint motor model (GO-M8010-6 specs; resistance estimated at 0.3 Ω)",
        transform=ax.transAxes, color=INK2, fontsize=9)
ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "torque_speed_curve.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, facecolor="white")
print("Saved", out)
