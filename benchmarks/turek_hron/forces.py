#!/usr/bin/env python3
"""Drag and lift of the Turek-Hron CFD3 test from a run's <name>_forces.csv.

The decks set reference_velocity 1 and reference_area 0.002, so the CSV's Cx
and Cy are the forces with Turek & Hron's fluid density 1000 (the solver runs
at density 1): C = 2 F / (U^2 A) = 1000 F. Prints, as Turek & Hron do, over
the last whole lift period: mean = (max + min) / 2, amplitude = (max - min) / 2,
and the frequency 1 / T (T from the lift's last upward zero crossings).

usage: forces.py FORCES.csv [--periods N]
"""
import argparse
import numpy as np

REF = {"drag": (439.45, 5.6183), "lift": (-11.893, 437.81), "frequency": 4.3956}

ap = argparse.ArgumentParser()
ap.add_argument("forces")
ap.add_argument("--periods", type=int, default=1,
                help="whole lift periods to evaluate over (default 1, the last)")
a = ap.parse_args()

d = np.genfromtxt(a.forces, delimiter=",", names=True)
t, drag, lift = d["t"], d["Cx"], d["Cy"]
# Upward crossings of the lift's running mean level, linearly interpolated.
level = np.median(lift[t > t[-1] - 2.0])
s = lift - level
up = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
tc = t[up] - s[up] * (t[up + 1] - t[up]) / (s[up + 1] - s[up])
if len(tc) < a.periods + 1:
    raise SystemExit("forces.py: not enough lift periods in the record")
t0, t1 = tc[-1 - a.periods], tc[-1]
m = (t >= t0) & (t <= t1)
freq = a.periods / (t1 - t0)
for name, y in (("drag", drag[m]), ("lift", lift[m])):
    mean, amp = 0.5 * (y.max() + y.min()), 0.5 * (y.max() - y.min())
    rm, ra = REF[name]
    print(f"{name}: {mean:10.4f} +- {amp:9.4f}   (Turek & Hron 2006: {rm} +- {ra}; "
          f"errors {100 * (mean - rm) / abs(rm):+.2f} % of the mean, "
          f"{100 * (amp - ra) / ra:+.2f} % of the amplitude)")
print(f"frequency: {freq:.4f}   (Turek & Hron: {REF['frequency']}; "
      f"{100 * (freq - REF['frequency']) / REF['frequency']:+.2f} %), "
      f"t in [{t0:.3f}, {t1:.3f}]")
