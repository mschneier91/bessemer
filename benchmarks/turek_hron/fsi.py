#!/usr/bin/env python3
"""The Turek-Hron FSI2 / FSI3 quantities from a run's CSVs.

Reads <name>_forces.csv (drag Cx and lift Cy: the decks' reference_area 0.002
makes them Turek & Hron's forces at fluid density 1000) and <name>_fsi.csv
(the tip point A's displacement), and prints, as Turek & Hron (2006) do, over
the last whole period: mean = (max + min) / 2, amplitude = (max - min) / 2 and
the frequency (u_y and lift oscillate at the flag's frequency, u_x and drag
at twice it). Displacements in units of 1e-3.

With --smooth W the amplitudes are also given after a centred moving average over
+- W time units: the forces carry a step-to-step ripple from the coupling's residual
(its RMS doubles when the step halves) whose peaks the max - min amplitude picks up.
W = 0.005 takes under 0.3 % off FSI2's fundamentals.

usage: fsi.py {fsi2,fsi3} FORCES.csv FSI.csv [--periods N] [--smooth W]
"""
import argparse
import numpy as np

# mean, amplitude, frequency (Turek & Hron 2006, Table 12/13, finest level).
REF = {
    "fsi2": {"u_x": (-14.58, 12.44, 3.8), "u_y": (1.23, 80.6, 2.0),
             "drag": (208.83, 73.75, 3.8), "lift": (0.88, 234.2, 2.0)},
    "fsi3": {"u_x": (-2.69, 2.53, 10.9), "u_y": (1.48, 34.38, 5.3),
             "drag": (457.3, 22.66, 10.9), "lift": (2.22, 149.78, 5.3)},
}

ap = argparse.ArgumentParser()
ap.add_argument("case", choices=sorted(REF))
ap.add_argument("forces")
ap.add_argument("fsi")
ap.add_argument("--periods", type=int, default=1,
                help="whole periods of u_y to evaluate over (default 1, the last)")
ap.add_argument("--smooth", type=float, default=0.0,
                help="also the amplitudes after a moving average over +- this time")
a = ap.parse_args()

f = np.genfromtxt(a.forces, delimiter=",", names=True)
s = np.genfromtxt(a.fsi, delimiter=",", names=True)
ts, ux, uy = s["t"], 1e3 * s["displacement_x"], 1e3 * s["displacement_y"]

# The period from u_y's upward crossings of its recent mean level.
level = np.median(uy[ts > ts[-1] - 2.0])
y = uy - level
up = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
tc = ts[up] - y[up] * (ts[up + 1] - ts[up]) / (y[up + 1] - y[up])
if len(tc) < a.periods + 1:
    raise SystemExit("fsi.py: not enough periods of u_y in the record")
t0, t1 = tc[-1 - a.periods], tc[-1]
freq = a.periods / (t1 - t0)


def stats(t, v):
    m = (t >= t0) & (t <= t1)
    return 0.5 * (v[m].max() + v[m].min()), 0.5 * (v[m].max() - v[m].min())


def smooth(t, v, w):
    k = max(1, int(round(w / np.median(np.diff(t)))))
    s = np.convolve(v, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    s[:k] = s[-k:] = np.nan  # no full window
    return s


rows = (("u_x", ts, ux, 2), ("u_y", ts, uy, 1), ("drag", f["t"], f["Cx"], 2),
        ("lift", f["t"], f["Cy"], 1))
print(f"{a.case}: t in [{t0:.3f}, {t1:.3f}], frequency {freq:.4f}")
for name, t, v, mult in rows:
    mean, amp = stats(t, v)
    rm, ra, rf = REF[a.case][name]
    print(f"{name:5s} {mean:10.4f} +- {amp:9.4f} [{mult * freq:6.3f}]   "
          f"Turek & Hron {rm} +- {ra} [{rf}]: amplitude {100 * (amp - ra) / ra:+.2f} %, "
          f"frequency {100 * (mult * freq - rf) / rf:+.2f} %")
    if a.smooth > 0:
        mean, amp = stats(t, smooth(t, v, a.smooth))
        print(f"{'':5s} {mean:10.4f} +- {amp:9.4f} smoothed (+- {a.smooth}): "
              f"amplitude {100 * (amp - ra) / ra:+.2f} %")
