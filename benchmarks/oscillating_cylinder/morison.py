#!/usr/bin/env python3
"""In-line force coefficients of the oscillating cylinder (Dutsch et al. 1998).

Reads a run's <name>_forces.csv (columns t, Fx, Fy, Cx, Cy; C = F / (0.5 U^2 D)
with U = D = rho = 1) and prints c_d and c_i two ways over the whole periods
from --from on:

  Fourier (Dutsch's definition): the fundamental C ~ a cos(wt) + b sin(wt);
  with x' = -U cos(wt), c_d = 3 pi a / (8 U^2), c_i = -2 b / (pi U w).
  Least squares: C = -c_d x'|x'| - (pi / 2) c_i x'' over the same window.

usage: morison.py FORCES.csv [--from T] [--f 0.2] [--kc 5]
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("forces")
ap.add_argument("--from", dest="t0", type=float, default=25.0,
                help="start of the window (default 25: the periodic state, periods 6 on)")
ap.add_argument("--f", type=float, default=0.2, help="frequency (U = D = 1)")
ap.add_argument("--kc", type=float, default=5.0, help="Keulegan-Carpenter number")
a = ap.parse_args()

d = np.genfromtxt(a.forces, delimiter=",", names=True)
w, U = 2 * np.pi * a.f, a.kc * a.f
A = U / w
t, C = d["t"], d["Cx"]
T = 1 / a.f
n = int(np.floor((t[-1] - a.t0) / T + 1e-6))
if n < 1:
    raise SystemExit("morison.py: less than one whole period after --from")
m = t >= t[-1] - n * T - 1e-9
t, C = t[m], C[m]
dt = np.gradient(t)

ca = 2 / (n * T) * np.sum(C * np.cos(w * t) * dt)
cb = 2 / (n * T) * np.sum(C * np.sin(w * t) * dt)
print(f"Fourier, last {n} periods: c_d = {3 * np.pi * ca / (8 * U * U):.4f}  "
      f"c_i = {-2 * cb / (np.pi * U * w):.4f}   (Dutsch et al. 1998: 2.09 / 1.45)")

xd = -A * w * np.cos(w * t)
xdd = A * w * w * np.sin(w * t)
M = np.column_stack([-xd * np.abs(xd), -(np.pi / 2) * xdd])
s = np.sqrt(dt)
(cd, ci), *_ = np.linalg.lstsq(M * s[:, None], C * s, rcond=None)
print(f"least squares, same window: c_d = {cd:.4f}  c_i = {ci:.4f}")
