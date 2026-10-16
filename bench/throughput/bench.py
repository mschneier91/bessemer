#!/usr/bin/env python3
"""Throughput benchmark runner (docs/throughput.md): runs a case at fixed
steps, collects step times, iterations, phase times and the cheap correctness
quantities into JSON, and compares two result files.

  bench.py run --label base --sizes 32 --warmup 20 --measure 50 --repeats 3
  bench.py run --label base-phases --sizes 32 --warmup 5 --measure 20 \
               --repeats 1 --profile sync
  bench.py run --label replay --sizes 32 --restart <checkpoint dir> \
               --warmup 2 --measure 10 --repeats 1 --set solver.initial_guess=project
  bench.py compare results/base.json results/change.json [--log "what"]

With --restart the run continues a checkpoint (the replay harness: the same
linear systems, re-solved with the options given) at the checkpoint's next
step size, fixed; --sizes must be the checkpoint's mesh.

Standard library only (the Spack view's python has no PyYAML or numpy).
"""

import argparse
import csv
import datetime
import json
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CASES = os.path.join(REPO, "bench", "cases")
NOISE = 0.04  # shared-hardware noise threshold (relative), the plan's 4 %

# Phases of a step: profiler regions (leaf names) summed by exclusive time.
PHASES = {
    "fgmres_glue": ["fgmres"],
    "velocity_block": ["velocity_block::apply"],
    "schur_amg": ["cc::lor_amg"],
    "schur_bmb": ["cc::bmb", "cc::gradient", "cc::divergence",
                  "cc::velocity_mass_inverse"],
    "schur_other": ["inner_cg", "cc_schur::apply", "schur_block::apply",
                    "cc::pressure_mass_inverse"],
    "rhs_elimination": ["rhs", "stokes_solver::solve_true"],
    "explicit_rhs": ["time_integrator::bdf_rhs", "time_integrator::convection",
                     "time_integrator::forcing"],
    "refresh": ["stokes_solver::refresh", "stokes_operator::momentum",
                "momentum::assemble", "momentum::form_system",
                "momentum::diagonal", "momentum::rest_diagonal",
                "stokes_solver::velocity_pc", "time_integrator::adopt_solver",
                "cc::weighted_mass_setup"],
    "diagnostics_io": ["case::diagnostics", "case::record_step", "case::forces"],
    # The coupled FGMRES's Gram-Schmidt (inside fgmres_glue before
    # 2026-10-16), and the Schur (Uzawa) outer method's pieces
    # (solver.outer: schur; docs/uzawa.md).
    "gram_schmidt": ["fgmres::orthogonalize"],
    "schur_velocity_solves": ["schur_solver::velocity_solve",
                              "schur_operator::velocity_solve"],
    "schur_glue": ["schur_solver", "schur_solver::cg", "schur_operator::apply"],
}


def git_state():
    try:
        head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short",
                                        "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", REPO, "status", "--short",
                                         "--untracked-files=no"], text=True)
        return head + ("+dirty" if dirty.strip() else "")
    except Exception:
        return "unknown"


def gpu_name():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name",
                                       "--format=csv,noheader"], text=True)
        return out.strip().splitlines()[0]
    except Exception:
        return "none"


def checkpoint_meta(path):
    """t and next_dt of a checkpoint (its meta.yaml, flat key: value lines)."""
    meta = {}
    for line in open(os.path.join(path, "meta.yaml")):
        if ":" in line and not line.lstrip().startswith("#"):
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return float(meta["t"]), float(meta["next_dt"])


def parse_sets(sets):
    """--set section.key=value overrides as {("section", "key"): "value"}
    (any depth; a bare key is a top-level one); a later --set of a key wins."""
    out = {}
    for s in sets:
        key, eq, value = s.partition("=")
        path = tuple(key.split("."))
        if not eq or not all(path):
            sys.exit("bench.py: --set %r is not section.key=value" % s)
        out[path] = value
    for p in out:
        for q in out:
            if len(q) > len(p) and q[:len(p)] == p:
                sys.exit("bench.py: --set %s conflicts with --set %s"
                         % (".".join(p), ".".join(q)))
    return out


def apply_sets(text, sets):
    """The deck text with the overrides of parse_sets() applied. A key the
    deck already sets has its line replaced: a second line would be a
    duplicate key, which bessemer rejects (YAML readers keep only one of the
    two). A key the deck lacks goes at the end of its section, and a section
    the deck lacks at the end of the deck. Block-style decks only (the case
    templates): one key per line, nesting by indentation."""
    todo = dict(sets)
    lines = text.rstrip("\n").split("\n")
    stack = []    # (indent, key) of the sections enclosing the current line
    maps = {}     # section path -> {"indent": of its keys, "last": line, "seq"}
    scalars = set()
    for i, line in enumerate(lines):
        body = line.lstrip(" ")
        if not body or body.startswith("#"):
            continue
        indent = len(line) - len(body)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        for d in range(1, len(stack) + 1):  # the line extends its sections
            maps[tuple(k for _, k in stack[:d])]["last"] = i
        if stack:
            parent = maps[tuple(k for _, k in stack)]
            parent["indent"] = parent["indent"] or indent
            parent["seq"] = parent["seq"] or body.startswith("-")
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*):(?:\s+(.*))?$", body)
        if not m:
            continue
        path = tuple(k for _, k in stack) + (m.group(1),)
        if not m.group(2) or m.group(2).startswith("#"):  # opens a section
            if path in todo:
                sys.exit("bench.py: --set %s: a section, not a key" % ".".join(path))
            stack.append((indent, m.group(1)))
            maps[path] = {"indent": 0, "last": i, "seq": False, "at": indent}
        else:
            scalars.add(path)
            if path in todo:
                lines[i] = " " * indent + "%s: %s" % (m.group(1), todo.pop(path))
    inserts = {}  # deepest existing section -> {key: value or {key: ...}}
    for path, value in todo.items():
        d = len(path) - 1
        while d and path[:d] not in maps:
            d -= 1
        base = path[:d]
        if path[:d + 1] in scalars:
            sys.exit("bench.py: --set %s: %s is a key, not a section"
                     % (".".join(path), ".".join(path[:d + 1])))
        if base and maps[base]["seq"]:
            sys.exit("bench.py: --set %s: %s is a list" % (".".join(path), ".".join(base)))
        tree = inserts.setdefault(base, {})
        for k in path[d:-1]:
            tree = tree.setdefault(k, {})
        tree[path[-1]] = value

    def render(tree, indent):
        out = []
        for k, v in tree.items():
            if isinstance(v, dict):
                out += [" " * indent + k + ":"] + render(v, indent + 2)
            else:
                out.append(" " * indent + "%s: %s" % (k, v))
        return out
    # Bottom-up, so that earlier line numbers stay valid; a section before a
    # subsection ending on the same line, so that the subsection's new keys
    # land inside it; new top-level keys last, at the end.
    for base in sorted((b for b in inserts if b),
                       key=lambda b: (-maps[b]["last"], len(b))):
        s = maps[base]
        lines[s["last"] + 1:s["last"] + 1] = render(inserts[base],
                                                    s["indent"] or s["at"] + 2)
    lines += render(inserts.get((), {}), 0)
    return "\n".join(lines) + "\n"


def write_deck(case, n, steps, path, profile, sets, restart=None):
    tmpl = open(os.path.join(CASES, case, "deck.yaml.in")).read()
    t0, dt = 0.0, 0.5 / n
    if restart:
        t0, dt = checkpoint_meta(restart)
    text = (tmpl.replace("{N}", str(n))
            .replace("{DT}", repr(dt))
            .replace("{TFINAL}", repr(t0 + dt * steps * (1.0 + 1e-9)))
            .replace("{STEPS}", str(steps))
            .replace("{PATH}", path)
            .replace("{PROFILE}", profile))
    overrides = parse_sets(sets)
    if restart:
        overrides[("restart",)] = os.path.abspath(restart)
    return apply_sets(text, overrides)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def phases_from_profile(path):
    """Exclusive time of the run's regions (the profile CSV: every region;
    the summary keeps only the top levels), grouped into PHASES."""
    excl = {}
    try:
        rows = list(csv.DictReader(l for l in open(path) if not l.startswith("#")))
    except OSError:
        return {}
    for r in rows:
        if "case::run" not in r["path"]:
            continue
        excl[r["region"]] = excl.get(r["region"], 0.0) + float(r["excl_max"])
    total = sum(excl.values())
    phases = {}
    for phase, leaves in PHASES.items():
        phases[phase] = sum(excl.pop(l, 0.0) for l in leaves)
    phases["other"] = sum(excl.values())
    if total > 0:
        return {k: {"seconds": v, "fraction": v / total} for k, v in phases.items()}
    return {}


def output_name(deck):
    """The output.name of a written deck (its CSV and JSON prefix; a --set
    may have changed the template's)."""
    m = re.search(r"^  name: (\S+)$", open(deck).read(), flags=re.M)
    return m.group(1) if m else "tgv"


def one_run(args, case, n, rep, outdir):
    steps = args.warmup + args.measure
    rundir = os.path.join(outdir, "%s_n%d_r%d" % (case, n, rep))
    os.makedirs(rundir, exist_ok=True)
    deck = os.path.join(rundir, "deck.yaml")
    with open(deck, "w") as f:
        f.write(write_deck(case, n, steps, rundir, args.profile, args.set,
                           args.restart))
    wrap = [w.replace("{DIR}", rundir) for w in args.wrap.split()]
    cmd = ["mpirun", "-np", str(args.np)] + wrap + [args.bin, deck]
    env = dict(os.environ, INCNS_DEVICE=args.device)
    t0 = time.time()
    with open(os.path.join(rundir, "log.txt"), "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                             timeout=args.timeout)
    wall = time.time() - t0
    rec = {"size": n, "repeat": rep, "exit": rc, "wall_s": wall, "dir": rundir}
    name = output_name(deck)
    try:
        hist = read_csv(os.path.join(rundir, name + "_history.csv"))
        summ = json.load(open(os.path.join(rundir, name + "_summary.json")))
        diag = read_csv(os.path.join(rundir, name + "_diagnostics.csv"))
    except Exception as e:
        rec["error"] = "outputs missing: %s" % e
        return rec
    meas = hist[args.warmup:args.warmup + args.measure]
    sw = [float(r["step_wall"]) for r in meas]
    it = [float(r["iterations"]) for r in meas]
    dofs = int(summ["mesh"]["velocity_dofs"]) + int(summ["mesh"]["pressure_dofs"])
    energy = [float(r["kinetic_energy"]) for r in diag]
    eps = [float(r["dissipation"]) for r in diag]
    div = [float(r["divergence"]) for r in diag]
    ts = [float(r["t"]) for r in diag]
    # Budget closure: dE/dt + eps over the run (relative to eps): exact
    # only up to the scheme's own dissipation, so compared to a baseline.
    clos = []
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        if dt > 0 and eps[i] > 0:
            clos.append(((energy[i] - energy[i - 1]) / dt + 0.5 * (eps[i] + eps[i - 1]))
                        / eps[i])
    rec.update({
        "steps": len(hist), "measured_steps": len(meas), "dofs": dofs,
        "velocity_dofs": int(summ["mesh"]["velocity_dofs"]),
        "step_wall_mean": statistics.mean(sw) if sw else None,
        "step_wall_median": statistics.median(sw) if sw else None,
        "iterations_mean": statistics.mean(it) if it else None,
        "dof_steps_per_s": dofs / statistics.mean(sw) if sw else None,
        "energy_final": energy[-1] if energy else None,
        "divergence_max": max(div) if div else None,
        "budget_closure_mean": statistics.mean(clos) if clos else None,
        "phases": phases_from_profile(os.path.join(rundir, name + "_profile.csv")),
        "setup_s": float(flatten(summ.get("profile", {}).get("regions", {})).get(
            "case::setup.seconds", 0.0)),
    })
    rec.update(solver_log(os.path.join(rundir, name + "_solver.csv"),
                          args.warmup, args.measure))
    return rec


def solver_log(path, warmup, measure):
    """Means over the measured steps of the per-step solver log
    (solver.log_interval 1, which study.sh sets; absent otherwise): the work inside the solve, the
    final coupled residual against its tolerance, and the device memory in
    use after the step (the largest over the ranks)."""
    try:
        rows = read_csv(path)
    except OSError:
        return {}
    rows = rows[warmup:warmup + measure]  # one row per step, as the history
    if not rows:
        return {}

    def mean(key):
        return statistics.mean(float(r[key]) for r in rows)

    out = {"solver": {
        "outer": rows[-1]["outer"],
        "converged_all": all(r["converged"] == "1" for r in rows),
        "residual_over_tolerance_max": max(
            float(r["final_residual"]) / float(r["tolerance"])
            for r in rows if float(r["tolerance"]) > 0) if any(
                float(r["tolerance"]) > 0 for r in rows) else None,
        "device_used_gb_max": max(float(r["device_used_gb"]) for r in rows),
    }}
    for key in ("outer_iterations", "vel_inner_iterations", "schur_applications",
                "uzawa_cycles", "uzawa_vel_solves", "uzawa_vel_iterations",
                "uzawa_schur_applications", "uzawa_schur_inner_iterations",
                "uzawa_schur_cap_hits", "solve_time"):
        out["solver"][key + "_mean"] = mean(key)
    return out


def cmd_run(args):
    outdir = os.path.join(args.outroot, args.label)
    os.makedirs(outdir, exist_ok=True)
    result = {"label": args.label, "git": git_state(), "gpu": gpu_name(),
              "date": datetime.datetime.now().isoformat(timespec="seconds"),
              "case": args.case, "np": args.np, "device": args.device,
              "restart": args.restart,
              "protocol": {"warmup": args.warmup, "measure": args.measure,
                           "repeats": args.repeats, "profile": args.profile},
              "sets": args.set, "sizes": {}}
    for n in args.sizes:
        runs = [one_run(args, args.case, n, r, outdir) for r in range(args.repeats)]
        ok = [r for r in runs if r.get("exit") == 0 and r.get("dof_steps_per_s")]
        med = {}
        if ok:
            for k in ("step_wall_mean", "iterations_mean", "dof_steps_per_s",
                      "energy_final", "divergence_max", "budget_closure_mean"):
                med[k] = statistics.median(r[k] for r in ok)
            med["dofs"] = ok[0]["dofs"]
            med["phases"] = ok[0]["phases"]
            med["spread"] = ((max(r["step_wall_mean"] for r in ok)
                              - min(r["step_wall_mean"] for r in ok))
                             / med["step_wall_mean"])
        result["sizes"][str(n)] = {"median": med, "runs": runs}
        line = "n%-3d" % n
        if med:
            line += (" dofs %10d  step %.4f s  iters %.2f  %.3e DOF-steps/s  spread %.1f%%"
                     % (med["dofs"], med["step_wall_mean"], med["iterations_mean"],
                        med["dof_steps_per_s"], 100 * med["spread"]))
        else:
            line += " FAILED: " + "; ".join(str(r.get("error", r.get("exit"))) for r in runs)
        print(line, flush=True)
    out = args.out or os.path.join(outdir, "result.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", out)


def cmd_compare(args):
    a = json.load(open(args.before))
    b = json.load(open(args.after))
    rows = []
    verdicts = []
    for n, sb in b["sizes"].items():
        sa = a["sizes"].get(n)
        if not sa or not sa["median"] or not sb["median"]:
            continue
        ma, mb = sa["median"], sb["median"]
        speed = mb["dof_steps_per_s"] / ma["dof_steps_per_s"] - 1.0
        dits = mb["iterations_mean"] - ma["iterations_mean"]
        de = abs(mb["energy_final"] - ma["energy_final"]) / abs(ma["energy_final"])
        dr = mb["divergence_max"] / max(ma["divergence_max"], 1e-300)
        verdict = ("faster" if speed > NOISE else "slower" if speed < -NOISE
                   else "within noise")
        verdicts.append(verdict)
        rows.append("| %s | n%s | %.3e -> %.3e | %+.1f%% | %.2f -> %.2f | %.1e | %.2f | %s |"
                    % (args.what or b["label"], n, ma["dof_steps_per_s"],
                       mb["dof_steps_per_s"], 100 * speed, ma["iterations_mean"],
                       mb["iterations_mean"], de, dr, verdict))
    head = ("| change | size | DOF-steps/s | speed | iters/step | dE/E | div ratio | verdict |\n"
            "|---|---|---|---|---|---|---|---|")
    print(head)
    print("\n".join(rows))
    if args.log:
        new = not os.path.exists(args.log)
        with open(args.log, "a") as f:
            if new:
                f.write("# Throughput decision log\n\n")
            f.write("\n### %s (%s vs %s, %s)\n\n%s\n%s\n" % (
                args.what or b["label"], b["git"], a["git"],
                datetime.date.today().isoformat(), head, "\n".join(rows)))
            if args.note:
                f.write("\n%s\n" % args.note)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--case", default="tgv_re1600")
    r.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")],
                   default=[32])
    r.add_argument("--warmup", type=int, default=20)
    r.add_argument("--measure", type=int, default=50)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--np", type=int, default=1)
    r.add_argument("--device", default="cuda")
    r.add_argument("--profile", default="wall", choices=["off", "wall", "sync"])
    r.add_argument("--set", action="append", default=[],
                   help="section.key=value, e.g. solver.rtol=1e-8: replaces the "
                        "case template's value of the key, or adds the key")
    r.add_argument("--bin", default=os.path.join(REPO, "build", "cuda", "apps",
                                                 "run_case"))
    r.add_argument("--outroot", default=os.path.join(
        os.environ.get("BENCH_OUT", os.path.join(REPO, "..", "runs", "bench"))))
    r.add_argument("--out")
    r.add_argument("--restart", help="checkpoint directory to continue")
    r.add_argument("--timeout", type=int, default=3600)
    r.add_argument("--wrap", default="",
                   help="command between mpirun and run_case, e.g. nsys profile -o {DIR}/nsys")
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--what")
    c.add_argument("--log")
    c.add_argument("--note")
    args = p.parse_args()
    {"run": cmd_run, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
