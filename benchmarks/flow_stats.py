#!/usr/bin/env python3
"""Normalize and reduce the flow statistics CSVs (deck section flow_statistics).

A run with flow_statistics writes <path>/<name>_stats_<set>.csv for every
sample set (lines, radial profiles, wall bins; docs/using.md section 5) in
the solver's units: the run's lengths and velocities, and the pressure and
stresses per unit density (kinematic). Standard library only.

  normalize CSV [--u-ref U] [--l-ref L] [--origin X0] [--p-ref P |
                --p-ref-csv CSV --p-ref-at X] [--p-factor F] [--stress-factor F]
                [--rho RHO] [-o OUT]
      One set's CSV in normalized units: the axial coordinate (s, station or
      xi) as (x - X0) / L and the other lengths / L, the velocities / U, the
      pressure as (p - p_ref) / (F U^2) (F = 0.5: the dynamic pressure), the
      Reynolds and viscous stresses and the wall shear stress / (F_s U^2)
      (F_s = 0.5). Without --u-ref, --rho makes the pressure and stresses
      dimensional instead (times rho). p_ref: a value, or the pressure of
      another set's CSV at axial coordinate X (linear interpolation; a
      profile's r = 0 rows).

  centerline CSV [--start S] [--fraction F]
      Where a jet breaks down along a line set (there is no single standard;
      these are the usual indicators): the peak of the velocity along the
      line, the first point after it where that velocity falls below F
      (default 0.5) of the peak, the steepest decrease after the peak, and
      the peak of the rms fluctuation along the line, sqrt(e.R.e).

  wall CSV
      The axial coordinates where wss_axial changes sign (separation and
      reattachment), by linear interpolation between bins.

  halves SUMMARY.json
      The run summary's half-window comparison as a table: max |first half -
      second half| over each set, relative to the set's scale.

  --selftest
      Checks the above on synthetic files.
"""
import argparse
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import interp, read_csv  # noqa: E402

AXIAL = ("s", "station", "xi", "xi_from", "xi_to")
LENGTH = ("x", "y", "z", "r")
VELOCITY = ("u", "v", "w", "u_a", "u_r", "u_t")


def kind_of(cols):
    for name, kind in (("s", "line"), ("station", "profile"), ("xi", "wall")):
        if name in cols:
            return kind
    raise ValueError("not a flow statistics CSV (no s, station or xi column)")


def read_header(path):
    """The '# ...' lines of a CSV, verbatim."""
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.startswith("#")]


def write_csv(path, cols, header):
    out = open(path, "w") if path else sys.stdout
    try:
        for h in header:
            out.write(h + "\n")
        names = list(cols)
        out.write(",".join(names) + "\n")
        for i in range(len(cols[names[0]])):
            out.write(",".join(repr(cols[n][i]) for n in names) + "\n")
    finally:
        if path:
            out.close()


def pressure_at(path, x):
    """The pressure of a set's CSV at axial coordinate x (a profile: r = 0)."""
    cols, _ = read_csv(path)
    kind = kind_of(cols)
    axial = {"line": "s", "profile": "station", "wall": "xi"}[kind]
    rows = [i for i in range(len(cols["p"]))
            if not math.isnan(cols["p"][i])
            and (kind != "profile" or cols["r"][i] == 0.0)]
    pairs = sorted((cols[axial][i], cols["p"][i]) for i in rows)
    p = interp(x, [a for a, _ in pairs], [b for _, b in pairs])
    if p is None:
        raise SystemExit("flow_stats.py: %s does not reach %s = %g" % (path, axial, x))
    return p


def normalize(cols, u_ref=None, l_ref=1.0, origin=0.0, p_ref=0.0,
              p_factor=0.5, stress_factor=0.5, rho=None):
    """The columns of a set in normalized units (see the module docs)."""
    dim = 3 if ("z" in cols or "u_a" in cols or "wss_cross" in cols) else 2
    if u_ref:
        p_scale = 1.0 / (p_factor * u_ref * u_ref)
        s_scale = 1.0 / (stress_factor * u_ref * u_ref)
        v_scale = 1.0 / u_ref
    else:
        p_scale = s_scale = rho if rho else 1.0
        v_scale = 1.0
    out = {}
    for name, vals in cols.items():
        if name in AXIAL:
            out[name] = [(v - origin) / l_ref for v in vals]
        elif name in LENGTH:
            out[name] = [v / l_ref for v in vals]
        elif name == "area":
            out[name] = [v / l_ref ** (dim - 1) for v in vals]
        elif name in VELOCITY:
            out[name] = [v * v_scale for v in vals]
        elif name == "p":
            out[name] = [(v - p_ref) * p_scale for v in vals]
        elif name.startswith(("R_", "tau_", "wss_")):
            out[name] = [v * s_scale for v in vals]
        else:
            out[name] = list(vals)
    return out


def line_direction(comments):
    def vec(key):
        v = comments[key]
        return [float(c) for c in (v.split() if isinstance(v, str) else [v])]
    a, b = vec("from"), vec("to")
    d = [y - x for x, y in zip(a, b)]
    n = math.sqrt(sum(c * c for c in d))
    return [c / n for c in d]


def centerline(cols, comments, start=None, fraction=0.5):
    """Jet breakdown indicators along a line set (see the module docs)."""
    e = line_direction(comments)
    ax = "xyz"[:len(e)]
    vel = "uvw"[:len(e)]
    s, uc, rms = [], [], []
    for i in range(len(cols["s"])):
        u = [cols[c][i] for c in vel]
        if any(math.isnan(x) for x in u) or (start is not None and cols["s"][i] < start):
            continue
        r = 0.0
        for a in range(len(e)):
            for b in range(len(e)):
                key = "R_" + "".join(sorted((ax[a], ax[b]), key=ax.index))
                r += e[a] * e[b] * cols[key][i]
        s.append(cols["s"][i])
        uc.append(sum(x * y for x, y in zip(e, u)))
        rms.append(math.sqrt(max(r, 0.0)))
    if len(s) < 3:
        raise SystemExit("flow_stats.py: too few points on the line")
    k = max(range(len(uc)), key=lambda i: uc[i])
    res = {"peak_s": s[k], "peak_velocity": uc[k]}
    level = fraction * uc[k]
    res["below_fraction_s"] = None
    for i in range(k, len(s) - 1):
        if uc[i] >= level > uc[i + 1]:
            w = (uc[i] - level) / (uc[i] - uc[i + 1])
            res["below_fraction_s"] = s[i] + w * (s[i + 1] - s[i])
            break
    best, where = 0.0, None
    for i in range(max(k, 1), len(s) - 1):
        slope = (uc[i + 1] - uc[i - 1]) / (s[i + 1] - s[i - 1])
        if slope < best:
            best, where = slope, s[i]
    res["steepest_decrease_s"] = where
    res["steepest_slope"] = best
    j = max(range(len(rms)), key=lambda i: rms[i])
    res["rms_peak_s"] = s[j]
    res["rms_peak"] = rms[j]
    return res


def sign_changes(cols, key="wss_axial", axial="xi"):
    """Axial coordinates where a column changes sign (linear interpolation)."""
    pts = [(x, v) for x, v in zip(cols[axial], cols[key]) if not math.isnan(v)]
    out = []
    for (x0, v0), (x1, v1) in zip(pts, pts[1:]):
        if v0 == 0.0:
            out.append(x0)
        elif v0 * v1 < 0.0:
            out.append(x0 + v0 / (v0 - v1) * (x1 - x0))
    return out


def halves(summary):
    st = summary.get("flow_statistics", {})
    rows = []
    for name, groups in sorted(st.get("half_window_difference", {}).items()):
        for g, v in sorted(groups.items()):
            rows.append((name, g, v))
    return st, rows


def selftest():
    fails = []

    def check(what, got, want, tol):
        if got is None or abs(got - want) > tol:
            fails.append("%s: %s, expected %s" % (what, got, want))

    with tempfile.TemporaryDirectory() as d:
        # A line along x: u falls from 1 to 0 around s = 5; rms peaks at s = 6.
        n = 201
        s = [10.0 * i / (n - 1) for i in range(n)]
        line = {"s": s, "x": s, "y": [0.0] * n, "z": [0.0] * n,
                "u": [1.0 / (1.0 + math.exp((x - 5.0) / 0.2)) for x in s],
                "v": [0.0] * n, "w": [0.0] * n, "p": [2.0 + x for x in s]}
        for k in ("xx", "yy", "zz", "xy", "xz", "yz"):
            line["R_" + k] = [math.exp(-(x - 6.0) ** 2) if k == "xx" else 0.0 for x in s]
            line["tau_" + k] = [0.1] * n
        path = os.path.join(d, "run_stats_center.csv")
        write_csv(path, line, ["# flow statistics (incns post/flow_sampling): line center",
                               "# from 0 0 0", "# to 10 0 0"])
        cols, comments = read_csv(path)
        r = centerline(cols, comments)
        check("peak", r["peak_s"], 0.0, 1e-12)
        check("below 0.5", r["below_fraction_s"], 5.0, 2e-3)
        check("steepest", r["steepest_decrease_s"], 5.0, 1e-9)
        check("rms peak", r["rms_peak_s"], 6.0, 1e-9)
        p_ref = pressure_at(path, 1.0)
        check("p_ref", p_ref, 3.0, 1e-12)
        nc = normalize(cols, u_ref=2.0, l_ref=2.0, origin=1.0, p_ref=p_ref)
        check("s'", nc["s"][100], (s[100] - 1.0) / 2.0, 1e-12)
        check("x'", nc["x"][100], s[100] / 2.0, 1e-12)
        check("p'", nc["p"][100], (s[100] - 1.0) / 2.0, 1e-12)
        check("u'", nc["u"][0], line["u"][0] / 2.0, 1e-12)
        check("tau'", nc["tau_xy"][0], 0.1 / 2.0, 1e-12)
        nd = normalize(cols, rho=1000.0)
        check("rho p", nd["p"][0], 2000.0, 1e-9)

        # A wall: wss_axial = cos(xi) on [0, 10].
        m = 400
        xi = [10.0 * (i + 0.5) / m for i in range(m)]
        wall = {"xi": xi, "xi_from": [x - 0.0125 for x in xi],
                "xi_to": [x + 0.0125 for x in xi], "area": [1.0] * m,
                "p": [0.0] * m, "wss_axial": [math.cos(x) for x in xi],
                "wss_cross": [0.0] * m}
        wpath = os.path.join(d, "run_stats_wall.csv")
        write_csv(wpath, wall, ["# flow statistics (incns post/flow_sampling): wall wall"])
        wc, _ = read_csv(wpath)
        z = sign_changes(wc)
        if len(z) != 3:
            fails.append("wall: %d sign changes, expected 3" % len(z))
        else:
            for k, x in enumerate(z):
                check("zero %d" % k, x, (k + 0.5) * math.pi, 1e-4)

        st, rows = halves({"flow_statistics": {"half_window_difference":
                                               {"center": {"velocity": 0.01}}}})
        if rows != [("center", "velocity", 0.01)]:
            fails.append("halves: %s" % rows)

    for f in fails:
        print("FAILED " + f)
    if not fails:
        print("flow_stats.py selftest ok")
    return 1 if fails else 0


def main():
    if "--selftest" in sys.argv[1:]:
        sys.exit(selftest())
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("normalize", help="a set's CSV in normalized units")
    n.add_argument("csv")
    n.add_argument("--u-ref", type=float)
    n.add_argument("--l-ref", type=float, default=1.0)
    n.add_argument("--origin", type=float, default=0.0,
                   help="where the axial coordinate (s, station, xi) is 0")
    n.add_argument("--p-ref", type=float, default=None)
    n.add_argument("--p-ref-csv")
    n.add_argument("--p-ref-at", type=float)
    n.add_argument("--p-factor", type=float, default=0.5)
    n.add_argument("--stress-factor", type=float, default=0.5)
    n.add_argument("--rho", type=float)
    n.add_argument("-o", "--output")
    c = sub.add_parser("centerline", help="jet breakdown indicators of a line")
    c.add_argument("csv")
    c.add_argument("--start", type=float)
    c.add_argument("--fraction", type=float, default=0.5)
    w = sub.add_parser("wall", help="sign changes of the axial wall shear stress")
    w.add_argument("csv")
    h = sub.add_parser("halves", help="the summary's half-window comparison")
    h.add_argument("summary")
    a = ap.parse_args()

    if a.cmd == "normalize":
        cols, _ = read_csv(a.csv)
        p_ref = a.p_ref if a.p_ref is not None else 0.0
        if a.p_ref_csv:
            if a.p_ref_at is None:
                raise SystemExit("flow_stats.py: --p-ref-csv needs --p-ref-at")
            p_ref = pressure_at(a.p_ref_csv, a.p_ref_at)
        out = normalize(cols, a.u_ref, a.l_ref, a.origin, p_ref, a.p_factor,
                        a.stress_factor, a.rho)
        header = read_header(a.csv) + [
            "# normalized by flow_stats.py: u_ref %s l_ref %g origin %g p_ref %.12g "
            "p_factor %g stress_factor %g rho %s" % (a.u_ref, a.l_ref, a.origin, p_ref,
                                                     a.p_factor, a.stress_factor, a.rho)]
        write_csv(a.output, out, header)
    elif a.cmd == "centerline":
        cols, comments = read_csv(a.csv)
        for k, v in centerline(cols, comments, a.start, a.fraction).items():
            print("%-22s %s" % (k, "none" if v is None else "%.10g" % v))
    elif a.cmd == "wall":
        cols, _ = read_csv(a.csv)
        for x in sign_changes(cols):
            print("%.10g" % x)
    else:
        with open(a.summary) as f:
            st, rows = halves(json.load(f))
        sp = st.get("split")
        if sp:
            print("split at t = %g: %g + %g averaging time" % (
                sp["time"], sp["first_averaging_time"], sp["second_averaging_time"]))
        else:
            print("no split yet (the window's midpoint has not passed)")
        for name, g, v in rows:
            print("%-20s %-16s %s" % (name, g, "n/a" if v is None else "%.3g %%" % (100 * v)))


if __name__ == "__main__":
    main()
