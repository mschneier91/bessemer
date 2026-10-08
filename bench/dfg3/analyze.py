#!/usr/bin/env python3
"""DFG 2D-3 scheme comparison: tables and work-precision summaries.

Reads the summary.txt files written by bench/dfg3/run_queue.sh (one line per
run: ``name=... kind=... RESULT key=value ...`` from apps/dfg_cylinder) and
prints, per scheme, every run's cost and its errors against John's reference
values, then the cheapest run reaching each accuracy level.

Usage: analyze.py SUMMARY [SUMMARY ...] [--json OUT.json]
"""

import json
import sys

# The errors the app reports (vs John 2004): relative for the peak values and
# dp(8), absolute (time units) for the peak times.
METRICS = {
    "err_cl": "c_L,max rel",
    "err_cd": "c_D,max rel",
    "err_tcl": "t(c_L,max) abs",
    "err_dp": "dp(8) rel",
}
LEVELS = {
    "err_cl": [1e-1, 3e-2, 1e-2, 3e-3, 1e-3],
    "err_cd": [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5],
    "err_tcl": [1e-1, 3e-2, 1e-2, 3e-3],
    "err_dp": [3e-2, 1e-2, 3e-3, 1e-3],
}


def parse(path):
    runs = []
    with open(path) as f:
        for line in f:
            if "RESULT" not in line:
                continue
            head, _, tail = line.partition("RESULT")
            r = {}
            for tok in (head + " " + tail).split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    try:
                        r[k] = float(v)
                    except ValueError:
                        r[k] = v
            r["source"] = path
            runs.append(r)
    return runs


def scheme(r):
    """Scheme label: form, extrapolation order and solver configuration."""
    form = r.get("form", "?")
    ext = int(r.get("ext", 0) or 0)
    if form == "convective":
        return "IMEX/EXT%d[%s]" % (ext, r.get("pc"))
    return "ROT/EXT%d[%s,%s]" % (ext, r.get("pc"), r.get("schur"))


def ok(r):
    return r.get("status") == "ok"


def main(argv):
    out_json = None
    if "--json" in argv:
        i = argv.index("--json")
        out_json = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    # Killed runs have no RESULT fields (their notes say why): skip them.
    runs = [r for p in argv for r in parse(p)
            if r.get("kind") == "run" and "form" in r]
    if not runs:
        print("no runs")
        return
    by = {}
    for r in runs:
        by.setdefault(scheme(r), []).append(r)

    print("%-28s %4s %5s %6s %7s %8s %6s %5s %6s | %9s %9s %8s %9s" % (
        "run", "mref", "cfl", "status", "steps", "step_wall", "s/step",
        "outer", "inner", "err_cL", "err_cD", "err_tcL", "err_dp"))
    for s in sorted(by):
        for r in sorted(by[s], key=lambda r: (r.get("mref", 0),
                                               r.get("cfl_target", 0))):
            st = r.get("steps", 0) or 1
            print("%-28s %4d %5.2f %6s %7d %8.1f %6.3f %5.1f %6.1f | %9.2e "
                  "%9.2e %8.1e %9.2e" % (
                      r["name"][:28], r.get("mref", 0), r.get("cfl_target", 0),
                      r.get("status"), r.get("steps", 0),
                      r.get("step_wall", float("nan")),
                      r.get("step_wall", float("nan")) / st,
                      r.get("outer_mean", 0), r.get("vel_inner_mean", 0),
                      r.get("err_cl", float("nan")), r.get("err_cd", float("nan")),
                      r.get("err_tcl", float("nan")), r.get("err_dp", float("nan"))))
        print()

    for key, label in METRICS.items():
        print("== %s: cheapest step_wall [s] reaching each level" % label)
        print("%-24s" % "scheme" + "".join("%11s" % ("<=%.0e" % e)
                                           for e in LEVELS[key]))
        for s in sorted(by):
            row = "%-24s" % s
            for eps in LEVELS[key]:
                c = [r["step_wall"] for r in by[s] if ok(r) and r[key] <= eps]
                row += "%11s" % ("%.0f" % min(c) if c else "-")
            print(row)
        print()

    if out_json:
        data = {s: [{k: v for k, v in r.items()} for r in rs]
                for s, rs in by.items()}
        with open(out_json, "w") as f:
            json.dump(data, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
