#!/usr/bin/env python3
"""Taylor-Green vortex, Re = 1600: compare runs with the spectral reference.

    python3 benchmarks/tgv_re1600/plot.py RUN_DIAGNOSTICS.csv [...] [-o out.png]
    python3 benchmarks/tgv_re1600/plot.py --selftest

Each argument is a run's <path>/<name>_diagnostics.csv (output.diagnostics):
t, kinetic_energy = int |u|^2 / 2, dissipation = nu int |grad u|^2, divergence.
Normalized by the box volume (2 pi)^3 these are the reference's E(t) and,
for this periodic flow, its eps(t) = -dE/dt; -dE/dt from the run's own E(t)
is shown as well (the two agree when the run is resolved).

Printed per run: the peak dissipation and its time against the reference's,
and the largest |eps - eps_ref| over the run, relative to the reference
peak. With matplotlib, also E(t) and eps(t) plots.
"""

import argparse
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common  # noqa: E402

VOLUME = (2.0 * math.pi) ** 3


def reference():
    cols, _ = common.read_reference(__file__, "reference", "spectral_re1600_512.csv")
    return cols


def load_run(path):
    cols, _ = common.read_csv(path)
    cols = common.last_per_time(cols)
    t = cols["t"]
    e = [k / VOLUME for k in cols["kinetic_energy"]]
    eps = [d / VOLUME for d in cols["dissipation"]]
    # -dE/dt by central differences (one-sided at the ends).
    dedt = []
    for i in range(len(t)):
        a, b = max(i - 1, 0), min(i + 1, len(t) - 1)
        dedt.append(-(e[b] - e[a]) / (t[b] - t[a]) if t[b] > t[a] else float("nan"))
    return {"t": t, "E": e, "eps": eps, "dEdt": dedt}


def compare(run, ref):
    ref_peak, ref_t = common.peak(ref["t"], ref["eps"])
    run_peak, run_t = common.peak(run["t"], run["eps"])
    worst = 0.0
    for t, eps in zip(run["t"], run["eps"]):
        r = common.interp(t, ref["t"], ref["eps"])
        if r is not None:
            worst = max(worst, abs(eps - r))
    return {"peak": run_peak, "t_peak": run_t, "ref_peak": ref_peak,
            "ref_t_peak": ref_t, "peak_rel_err": abs(run_peak - ref_peak) / ref_peak,
            "max_err_rel_peak": worst / ref_peak, "t_end": run["t"][-1]}


def report(name, m):
    print("%-28s eps_max %.6f at t %.3f  (ref %.6f at %.3f): peak error %.2f%%, "
          "max |eps - ref| %.2f%% of the peak (t <= %.2f)"
          % (name, m["peak"], m["t_peak"], m["ref_peak"], m["ref_t_peak"],
             100 * m["peak_rel_err"], 100 * m["max_err_rel_peak"], m["t_end"]))


def plot(runs, ref, out):
    plt = common.pyplot()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(ref["t"], ref["E"], "k-", lw=2, label="spectral 512^3")
    ax[1].plot(ref["t"], ref["eps"], "k-", lw=2, label="spectral 512^3")
    for name, r in runs:
        ax[0].plot(r["t"], r["E"], label=name)
        ax[1].plot(r["t"], r["eps"], label=name + ": nu<|grad u|^2>")
        ax[1].plot(r["t"], r["dEdt"], "--", label=name + ": -dE/dt")
    ax[0].set(xlabel="t", ylabel="E", title="Kinetic energy")
    ax[1].set(xlabel="t", ylabel="eps", title="Dissipation rate")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


def selftest(ref):
    """The reference itself, written as a run's diagnostics: zero error."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ref_diagnostics.csv")
        with open(path, "w") as f:
            f.write("t,kinetic_energy,dissipation,divergence\n")
            for t, e, eps in zip(ref["t"], ref["E"], ref["eps"]):
                f.write("%.17g,%.17g,%.17g,0\n" % (t, e * VOLUME, eps * VOLUME))
            # A restart's repeated row must not matter.
            f.write("%.17g,%.17g,%.17g,0\n"
                    % (ref["t"][5], ref["E"][5] * VOLUME, ref["eps"][5] * VOLUME))
        m = compare(load_run(path), ref)
    report("selftest (reference as run)", m)
    ok = m["peak_rel_err"] < 1e-9 and m["max_err_rel_peak"] < 1e-9 \
        and abs(m["ref_peak"] - 0.01286) < 5e-5 and abs(m["ref_t_peak"] - 8.97) < 0.05
    print("selftest", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", help="<name>_diagnostics.csv files")
    ap.add_argument("-o", "--out", default="tgv_re1600.png")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    ref = reference()
    if a.selftest:
        return selftest(ref)
    if not a.runs:
        ap.error("give at least one diagnostics CSV (or --selftest)")
    runs = []
    for p in a.runs:
        name = os.path.basename(os.path.dirname(os.path.abspath(p))) or p
        r = load_run(p)
        report(name, compare(r, ref))
        runs.append((name, r))
    plot(runs, ref, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
