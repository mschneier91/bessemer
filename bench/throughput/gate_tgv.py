#!/usr/bin/env python3
"""Expensive gate for Taylor-Green (docs/throughput.md): compare a run's
dissipation history with a reference run's.

  gate_tgv.py <reference diagnostics.csv> <run diagnostics.csv> [--tol 0.02]

Both CSVs are run_case diagnostics (t, kinetic_energy, dissipation,
divergence). Reports the peak dissipation and its time for each, the largest
difference of the dissipation curves (the run's interpolated to the
reference's times) relative to the reference peak, the final energies, and
the largest divergence norms; exits 1 when the curve difference exceeds
--tol. Standard library only.
"""

import argparse
import bisect
import csv
import sys


def load(path):
    rows = list(csv.DictReader(open(path)))
    t = [float(r["t"]) for r in rows]
    e = [float(r["kinetic_energy"]) for r in rows]
    eps = [float(r["dissipation"]) for r in rows]
    div = [float(r["divergence"]) for r in rows]
    return t, e, eps, div


def interp(ts, ys, t):
    i = bisect.bisect_left(ts, t)
    if i <= 0:
        return ys[0]
    if i >= len(ts):
        return ys[-1]
    w = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return (1 - w) * ys[i - 1] + w * ys[i]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reference")
    p.add_argument("run")
    p.add_argument("--tol", type=float, default=0.02)
    a = p.parse_args()
    tr, er, epr, dr = load(a.reference)
    tn, en, epn, dn = load(a.run)
    t_end = min(tr[-1], tn[-1])
    ipr = max(range(len(epr)), key=lambda i: epr[i])
    ipn = max(range(len(epn)), key=lambda i: epn[i])
    worst = max(abs(interp(tn, epn, t) - e) for t, e in zip(tr, epr) if t <= t_end)
    rel = worst / epr[ipr]
    e_ref, e_run = interp(tr, er, t_end), interp(tn, en, t_end)
    print("reference: peak dissipation %.6g at t = %.3f, E(%.2f) = %.8g, max div %.3g"
          % (epr[ipr], tr[ipr], t_end, e_ref, max(dr)))
    print("run:       peak dissipation %.6g at t = %.3f, E(%.2f) = %.8g, max div %.3g"
          % (epn[ipn], tn[ipn], t_end, e_run, max(dn)))
    print("largest dissipation difference over t <= %.2f: %.3e of the reference peak; "
          "final energy differs by %.3e (relative)" % (t_end, rel, abs(e_run - e_ref) / e_ref))
    ok = rel <= a.tol
    print("GATE " + ("PASS" if ok else "FAIL") + " (tolerance %.3g)" % a.tol)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
