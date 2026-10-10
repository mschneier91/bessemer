#!/usr/bin/env python3
"""DFG 3D flow around a cylinder: compare runs with the reference values.

    python3 benchmarks/dfg_3d/plot.py 3z RUN_HISTORY.csv [-o out.png]
    python3 benchmarks/dfg_3d/plot.py 1z RUN_SUMMARY.json
    python3 benchmarks/dfg_3d/plot.py --selftest

3z (unsteady, Re(t) up to 100): RUN_HISTORY.csv is the run's
<path>/<name>_history.csv (output.history with forces.statistics: c_d, c_l
every step). Compared with FeatFlow's finest drag and lift histories
(Bayraktar, Mierka & Turek 2012): c_D,max and c_L,min with their times, and
the largest differences over t in [0, 8]. With matplotlib, also both curves.

1z (steady, Re 20): RUN_SUMMARY.json is the run's summary; its final c_D,
c_L are compared with c_D = 6.18533, c_L = 0.009401.
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common  # noqa: E402

REF_1Z = {"cd": 6.18533, "cl": 0.009401}
REF_3Z = {"cd_max": 3.2978, "cl_min": -0.010999}  # FeatFlow, finest level


def reference_3z():
    cols, _ = common.read_reference(__file__, "reference", "featflow_3d3z.csv")
    return cols


def compare_3z(run, ref):
    cd_max, t_cd = common.peak(run["t"], run["c_d"])
    cl_min, t_cl = common.peak(run["t"], run["c_l"], find_min=True)
    rcd, rt_cd = common.peak(ref["t"], ref["c_d"])
    rcl, rt_cl = common.peak(ref["t"], ref["c_l"], find_min=True)
    dcd = dcl = 0.0
    for t, cd, cl in zip(run["t"], run["c_d"], run["c_l"]):
        a = common.interp(t, ref["t"], ref["c_d"])
        b = common.interp(t, ref["t"], ref["c_l"])
        if a is not None:
            dcd = max(dcd, abs(cd - a))
            dcl = max(dcl, abs(cl - b))
    return {"cd_max": cd_max, "t_cd_max": t_cd, "cl_min": cl_min, "t_cl_min": t_cl,
            "ref_cd_max": rcd, "ref_t_cd_max": rt_cd, "ref_cl_min": rcl,
            "ref_t_cl_min": rt_cl, "max_dcd": dcd, "max_dcl": dcl}


def report_3z(name, m):
    print("%s: c_D,max %.5f at t %.4f (FeatFlow %.5f at %.4f; tabulated %.4f), "
          "error %.3f%%" % (name, m["cd_max"], m["t_cd_max"], m["ref_cd_max"],
                            m["ref_t_cd_max"], REF_3Z["cd_max"],
                            100 * abs(m["cd_max"] - REF_3Z["cd_max"]) / REF_3Z["cd_max"]))
    print("  c_L,min %.6f at t %.4f (FeatFlow %.6f at %.4f; tabulated %.6f), error "
          "%.2f%%; max |c_D - ref| %.2e, max |c_L - ref| %.2e"
          % (m["cl_min"], m["t_cl_min"], m["ref_cl_min"], m["ref_t_cl_min"],
             REF_3Z["cl_min"],
             100 * abs(m["cl_min"] - REF_3Z["cl_min"]) / abs(REF_3Z["cl_min"]),
             m["max_dcd"], m["max_dcl"]))


def plot_3z(name, run, ref, out):
    plt = common.pyplot()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, key, title in ((ax[0], "c_d", "Drag"), (ax[1], "c_l", "Lift")):
        a.plot(ref["t"], ref[key], "k-", lw=2, label="FeatFlow (finest)")
        a.plot(run["t"], run[key], "--", label=name)
        a.set(xlabel="t", ylabel=key, title=title)
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


def selftest():
    ref = reference_3z()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dfg_history.csv")
        with open(path, "w") as f:
            f.write("t,dt,c_d,c_l,iterations,substeps,elements,step_wall\n")
            for t, cd, cl in zip(ref["t"], ref["c_d"], ref["c_l"]):
                f.write("%.17g,0.005,%.17g,%.17g,10,0,100,0.1\n" % (t, cd, cl))
        cols, _ = common.read_csv(path)
        m = compare_3z(common.last_per_time(cols), ref)
    report_3z("selftest (reference as run)", m)
    ok = (m["max_dcd"] < 1e-12 and m["max_dcl"] < 1e-12
          and abs(m["ref_cd_max"] - REF_3Z["cd_max"]) < 2e-3
          and abs(m["ref_cl_min"] - REF_3Z["cl_min"]) < 2e-4)
    print("selftest", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case", nargs="?", choices=["1z", "3z"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("-o", "--out", default="dfg_3d3z.png")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.case or not a.file:
        ap.error("give the case (1z or 3z) and the run's file (or --selftest)")
    if a.case == "1z":
        s = common.load_summary(a.file)
        f = s.get("forces", {})
        for key in ("cd", "cl"):
            v = f.get(key)
            print("%s: %.6f (reference %.6f), error %.3f%%"
                  % (key, v, REF_1Z[key], 100 * abs(v - REF_1Z[key]) / abs(REF_1Z[key])))
        return 0
    cols, _ = common.read_csv(a.file)
    run = common.last_per_time(cols)
    ref = reference_3z()
    name = os.path.basename(a.file)
    report_3z(name, compare_3z(run, ref))
    plot_3z(name, run, ref, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
