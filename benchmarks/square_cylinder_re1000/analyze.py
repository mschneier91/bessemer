#!/usr/bin/env python3
"""Compare convergence levels of the Re = 1000 square-cylinder benchmark.

usage:
    analyze.py [options] HISTORY.csv [HISTORY.csv ...]
    analyze.py [options] name=HISTORY.csv name=HISTORY.csv ...

Give the per-step history files (``output.history: true``) in order of increasing
resolution. For each: the average of C_D, the rms of C_L and the spectral peaks of C_L
over t >= --t0, each with a batch-means sampling error; then the difference between
consecutive levels against that error (the criterion of
docs/design/re1000_convergence_plan.md: a difference counts as real only if it exceeds
both the sampling error and --tol of the value).

Needs numpy (it is in environments/stack.yaml). --save DIR also writes, per level,
DIR/<name>_series.csv (t, c_d, c_l resampled to dt = 0.05) and DIR/<name>_stats.json.

The flow is irregular, not periodic: the deck's period finder is not used. The Strouhal
number is the position of the spectral peak of C_L; the two strongest peaks are printed
because the spectrum has two comparable ones (~0.10 and ~0.19 in the first runs).
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("analyze.py needs numpy (environments/stack.yaml provides it)")

H = 0.05  # uniform resampling step; Nyquist 10 is far above every frequency of interest


def load(path):
    t, cd, cl, el, dt, wall = [], [], [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            t.append(float(r["t"])); cd.append(float(r["c_d"])); cl.append(float(r["c_l"]))
            el.append(int(r["elements"])); dt.append(float(r["dt"])); wall.append(float(r["step_wall"]))
    return [np.array(x) for x in (t, cd, cl, el, dt, wall)]


def batch_stat(x, k, fn):
    """fn over batches of k samples: (value over everything, batch values)."""
    n = len(x) // k
    vals = np.array([fn(x[i * k:(i + 1) * k]) for i in range(n)])
    return vals


def spectral_peaks(y, h, fmin=0.03, fmax=0.5, count=2):
    y = y - y.mean()
    F = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), h)
    idx = [i for i in range(1, len(F) - 1)
           if F[i] > F[i - 1] and F[i] >= F[i + 1] and fmin <= f[i] <= fmax]
    idx.sort(key=lambda i: -F[i])
    out = []
    for i in idx:
        if all(abs(f[i] - f[j]) > 0.03 for j in out):
            out.append(i)
        if len(out) == count:
            break
    return [(float(f[i]), float(F[i] / F[(f >= fmin) & (f <= fmax)].sum())) for i in out], 1.0 / (len(y) * h)


def analyse(name, path, t0, batch):
    t, cd, cl, el, dt, wall = load(path)
    g = np.arange(0.0, t[-1], H)
    d, l = np.interp(g, t, cd), np.interp(g, t, cl)
    w = g >= t0
    if w.sum() < 3:  # the run ends before the window starts
        return {"name": name, "history": path, "t_end": float(t[-1]), "window": None,
                "window_length": 0.0, "batch": batch, "batches": 0, "steps": int(len(t)),
                "elements_final": int(el[-1]), "elements_max": int(el.max()),
                "dt_median": float(np.median(dt)), "wall_seconds": float(wall.sum()),
                "too_short": True, "spectral_peaks": [], "frequency_resolution": None,
                "st_crossing_rate": None}, (g, d, l)
    gw, dw, lw = g[w], d[w], l[w]
    T = gw[-1] - gw[0]
    k = int(round(batch / H))
    lmean = lw.mean()
    bd = batch_stat(dw, k, np.mean)
    bl = batch_stat(lw - lmean, k, lambda a: math.sqrt(np.mean(a ** 2)))
    n = len(bd)
    res = {
        "name": name, "history": path, "t_end": float(t[-1]), "window": [float(gw[0]), float(gw[-1])],
        "window_length": float(T), "batch": batch, "batches": n,
        "steps": int(len(t)), "elements_final": int(el[-1]), "elements_max": int(el.max()),
        "dt_median": float(np.median(dt)), "dt_min_after_start": float(dt[t > 5].min()) if (t > 5).any() else None,
        "wall_seconds": float(wall.sum()),
        "cd_mean": float(dw.mean()), "cl_mean": float(lmean),
        "cl_rms": float(math.sqrt(np.mean((lw - lmean) ** 2))),
    }
    if n >= 3:
        res["cd_mean_err"] = float(bd.std(ddof=1) / math.sqrt(n))
        res["cl_rms_err"] = float(bl.std(ddof=1) / math.sqrt(n))
        res["cd_batch_lag1"] = float(np.corrcoef(bd[:-1], bd[1:])[0, 1])
    res["too_short"] = bool(T < 2 * batch or n < 2)
    if res["too_short"]:
        res["spectral_peaks"], res["frequency_resolution"], res["st_crossing_rate"] = [], None, None
        return res, (g, d, l)
    peaks, df = spectral_peaks(lw, H)
    res["spectral_peaks"] = [{"f": f, "power_fraction": p} for f, p in peaks]
    res["frequency_resolution"] = df
    up = gw[1:][((lw - lmean)[:-1] < 0) & ((lw - lmean)[1:] >= 0)]
    res["st_crossing_rate"] = float(len(up) / T)
    return res, (g, d, l)


def fmt_err(v, e):
    return f"{v:.3f} +- {e:.3f}" if e is not None else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--t0", type=float, default=30.0, help="start of the averaging window (default 30)")
    ap.add_argument("--batch", type=float, default=10.0, help="batch length in time units (default 10)")
    ap.add_argument("--tol", type=float, default=0.02, help="agreement criterion as a fraction (default 0.02)")
    ap.add_argument("--save", metavar="DIR", help="write <name>_series.csv and <name>_stats.json here")
    a = ap.parse_args()

    levels = []
    for item in a.files:
        if "=" in item and not os.path.exists(item):
            name, path = item.split("=", 1)
        else:
            path = item
            name = os.path.basename(os.path.dirname(os.path.abspath(path))) or os.path.basename(path)
            base = os.path.basename(path)
            if base.endswith("_history.csv"):
                name = base[: -len("_history.csv")]
        levels.append((name, path))

    results, series = [], {}
    for name, path in levels:
        r, s = analyse(name, path, a.t0, a.batch)
        results.append(r); series[name] = s

    print(f"window t >= {a.t0:g}, batches of {a.batch:g} time units, criterion {100*a.tol:g} %\n")
    head = f"{'level':26s} {'elements':>9s} {'steps':>6s} {'dt med':>7s} {'wall h':>7s} {'window':>7s}  {'C_D mean':>16s}  {'C_L rms':>16s}  St peaks (power)"
    print(head)
    for r in results:
        if r["too_short"]:
            print(f"{r['name'][:26]:26s} {r['elements_final']:9d} {r['steps']:6d} {r['dt_median']:7.4f} "
                  f"{r['wall_seconds']/3600:7.2f} {r['window_length']:7.0f}  window too short for statistics "
                  f"(need >= {2*a.batch:g} time units from t0 = {a.t0:g}; run reaches t = {r['t_end']:.0f})")
            continue
        pk = ", ".join(f"{p['f']:.3f} ({100*p['power_fraction']:.0f}%)" for p in r["spectral_peaks"])
        print(f"{r['name'][:26]:26s} {r['elements_final']:9d} {r['steps']:6d} {r['dt_median']:7.4f} "
              f"{r['wall_seconds']/3600:7.2f} {r['window_length']:7.0f}  "
              f"{fmt_err(r['cd_mean'], r.get('cd_mean_err')):>16s}  "
              f"{fmt_err(r['cl_rms'], r.get('cl_rms_err')):>16s}  {pk}  [res {r['frequency_resolution']:.3f}, "
              f"crossings {r['st_crossing_rate']:.3f}]")
    for r in results:
        if r.get("cd_batch_lag1") is not None and abs(r["cd_batch_lag1"]) > 0.5:
            print(f"  note: {r['name']}: batch means are correlated (lag-1 {r['cd_batch_lag1']:+.2f}); "
                  f"the error bar is unreliable, lengthen the window or the batch")
        if r["batches"] < 5 and not r["too_short"]:
            print(f"  note: {r['name']}: only {r['batches']} batches; the error bar is crude")

    if len(results) > 1:
        print("\nconsecutive levels (a difference counts only above both the sampling error and the criterion):")
        for p, q in zip(results[:-1], results[1:]):
            if p["too_short"] or q["too_short"]:
                print(f"  {p['name']} -> {q['name']}  not compared: a window is too short")
                continue
            for key, errkey, lab in (("cd_mean", "cd_mean_err", "C_D mean"), ("cl_rms", "cl_rms_err", "C_L rms")):
                dv = q[key] - p[key]
                rel = dv / abs(p[key])
                e = math.hypot(p.get(errkey) or 0.0, q.get(errkey) or 0.0)
                real = abs(dv) > max(2 * e, a.tol * abs(p[key]))
                print(f"  {p['name']} -> {q['name']}  {lab:9s} {dv:+.3f} ({100*rel:+.1f} %)  "
                      f"2 sigma = {2*e:.3f} ({100*2*e/abs(p[key]):.1f} %)  "
                      f"{'DIFFERS' if real else 'within error / criterion'}")
            if not p["spectral_peaks"] or not q["spectral_peaks"]:
                continue
            f1p, f1q = p["spectral_peaks"][0]["f"], q["spectral_peaks"][0]["f"]
            res = max(p["frequency_resolution"], q["frequency_resolution"])
            print(f"  {p['name']} -> {q['name']}  main peak {f1p:.3f} -> {f1q:.3f}  "
                  f"({'within' if abs(f1q - f1p) <= res else 'beyond'} the frequency resolution {res:.3f})")

    if a.save:
        os.makedirs(a.save, exist_ok=True)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                             cwd=os.path.dirname(os.path.abspath(__file__))).strip()
        except Exception:
            commit = None
        for r in results:
            g, d, l = series[r["name"]]
            with open(os.path.join(a.save, f"{r['name']}_series.csv"), "w") as f:
                f.write("# Re 1000 square cylinder, resampled history (linear interpolation, dt = 0.05)\n")
                f.write("t,c_d,c_l\n")
                for x, y, z in zip(g, d, l):
                    f.write(f"{x:.2f},{y:.5f},{z:.5f}\n")
            r2 = dict(r, analysis_t0=a.t0, analysis_commit=commit)
            with open(os.path.join(a.save, f"{r['name']}_stats.json"), "w") as f:
                json.dump(r2, f, indent=1)
        print(f"\nwrote series and stats for {len(results)} level(s) to {a.save}")


if __name__ == "__main__":
    main()
