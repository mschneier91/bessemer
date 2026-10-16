#!/usr/bin/env python3
"""Compare a coupled and a Schur (Uzawa) run of bench/throughput/bench.py
(docs/uzawa.md, the study): per size, the step time, the iterations and the
work inside the solve, the final coupled residual against its tolerance, the
device memory in use, and the correctness quantities (final kinetic energy,
largest divergence norm).

  compare.py results/uzawa/coupled/result.json results/uzawa/schur/result.json

The decision rule of the proposal (docs/design/uzawa_schur_cg_spec.md, s. 7):
Schur becomes the default only if its step is within 1.2x of coupled at a
matched coupled residual AND its peak memory is at least 25 % lower. The
memory here is what the per-step log saw after each step (the solver's
workspace is held across steps); the peak inside a step needs
--profile sync (the profile CSV's gauge columns).

Standard library only.
"""

import json
import sys


def median(result, n):
    s = result["sizes"].get(n)
    return s["median"] if s and s["median"] else None


def first_solver(result, n):
    runs = result["sizes"][n]["runs"]
    ok = [r for r in runs if r.get("exit") == 0 and "solver" in r]
    return ok[0]["solver"] if ok else {}


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))
    print("%s (%s) vs %s (%s)" % (a["label"], a["git"], b["label"], b["git"]))
    for n in sorted(b["sizes"], key=int):
        ma, mb = median(a, n), median(b, n)
        if not ma or not mb:
            print("n%s: a run failed" % n)
            continue
        sa, sb = first_solver(a, n), first_solver(b, n)
        ratio = mb["step_wall_mean"] / ma["step_wall_mean"]
        de = abs(mb["energy_final"] - ma["energy_final"]) / abs(ma["energy_final"])
        print("n%s  dofs %d" % (n, mb["dofs"]))
        print("  step        %.4f s -> %.4f s  (x%.3f)" %
              (ma["step_wall_mean"], mb["step_wall_mean"], ratio))
        print("  outer its   %.2f -> %.2f" %
              (ma["iterations_mean"], mb["iterations_mean"]))
        print("  energy      rel. diff %.2e   divergence %.3e -> %.3e" %
              (de, ma["divergence_max"], mb["divergence_max"]))
        if sa and sb:
            ga, gb = sa["device_used_gb_max"], sb["device_used_gb_max"]
            print("  memory      %.2f GB -> %.2f GB  (%+.1f %%)" %
                  (ga, gb, 100 * (gb / ga - 1) if ga > 0 else float("nan")))
            print("  residual/tol max  %s -> %s   converged %s / %s" %
                  (sa["residual_over_tolerance_max"],
                   sb["residual_over_tolerance_max"], sa["converged_all"],
                   sb["converged_all"]))
            print("  coupled velocity-PC inner its/step %.1f" %
                  sa["vel_inner_iterations_mean"])
            print("  schur: velocity solves %.1f, their its %.1f, S applications"
                  " %.1f, cycles %.2f, cap hits %.2f per step" %
                  (sb["uzawa_vel_solves_mean"], sb["uzawa_vel_iterations_mean"],
                   sb["uzawa_schur_applications_mean"],
                   sb["uzawa_cycles_mean"], sb["uzawa_schur_cap_hits_mean"]))
            ok_time = ratio <= 1.2
            ok_mem = ga > 0 and gb <= 0.75 * ga
            print("  decision rule: time %s, memory %s" %
                  ("met" if ok_time else "NOT met",
                   "met" if ok_mem else "NOT met"))
        else:
            print("  (no solver log: run with --set solver.log_interval=1)")


if __name__ == "__main__":
    main()
