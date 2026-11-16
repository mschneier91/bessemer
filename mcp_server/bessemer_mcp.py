#!/usr/bin/env python3
"""bessemer_mcp -- an MCP server that lets an AI assistant set up, vet, run and
read bessemer cases on the user's own machine.

Standard library only (Python >= 3.8): nothing to install. It speaks the Model
Context Protocol over stdio (newline-delimited JSON-RPC 2.0) and drives the
user's own build of `run_case`:

  * `run_case --deck-schema`   every deck key, so the key list never goes stale;
  * `run_case DECK --check`    load the deck, build the mesh, check the
                               boundary groups, count the DOFs; no solve;
  * `run_case DECK`            the run itself, in the background.

Nothing here submits to a scheduler or holds credentials. How to connect it to
an assistant, and what each tool does: docs/assistant.md.

Environment (all optional):
  BESSEMER_ROOT      the repository (default: this file's parent's parent)
  BESSEMER_RUN_CASE  the run_case binary (default: build/<preset>/apps/run_case)
  BESSEMER_ENV       auto | spack | none: run through scripts/env.sh (spack),
                     plainly (none), or spack when env.sh works (auto, default)
  BESSEMER_MPIEXEC   the MPI launcher when not through env.sh (default mpirun)
  BESSEMER_RUNS      where new cases and the run registry go (default runs/)
"""

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

SERVER_NAME = "bessemer"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]

ROOT = os.path.abspath(os.environ.get(
    "BESSEMER_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
RUNS = os.path.abspath(os.environ.get("BESSEMER_RUNS", os.path.join(ROOT, "runs")))
REGISTRY = os.path.join(RUNS, ".mcp")

# Memory per velocity DOF, for the estimate check_deck gives. Measured on one
# H100 (docs/gpu_memory.md, docs/status.md): about 0.6-0.9 kB per velocity
# DOF at Q6/Q5 depending on the solver path, about 1.5 kB at Q3/Q2. A CPU run
# holds the same data in host memory. An estimate, not a bound: AMR grows the
# mesh during the run, and the FSI solid and checkpoints add to it.
BYTES_PER_VDOF = (600.0, 1500.0)

INSTRUCTIONS = """\
bessemer is a high-order finite element solver for incompressible flow
(Navier-Stokes, Stokes), linear and hyperelastic solids, mesh motion and
fluid-structure interaction, on quad/hex meshes, MPI-parallel. Cases are YAML
decks. This server runs on the user's own machine.

How to help a user (they know the flow they want, not this code base):
1. check_setup first. If run_case is missing, show them `scripts/setup.sh
   --plan` (read docs with read_doc "install") and run nothing until they agree.
2. Turn the physical request into a case: list_examples, start from the
   closest deck with new_case (it copies the deck into runs/<name>/, keeps its
   comments, applies dotted-key changes). Look up keys with deck_keys; never
   assume a default, read it from deck_keys. Set the physics they asked for,
   pick resolution, orders, time stepping and end time yourself, and state those
   choices in a sentence or two. Ask only about choices that change the answer
   or the cost a lot.
3. check_deck before every run. Fix the errors it reports (it gives the
   loader's own message). Tell the user the size (DOFs) and memory estimate.
4. Say the cost before a long run. Start it with run_deck; after the first
   progress lines run_status gives an ETA from the run itself. If that is over
   ~10 minutes, tell the user and ask before letting it continue.
5. Report from read_results (summary.json), never from the log: the quantities
   asked for, and the reference errors when the deck has reference values. Say
   what one run does not establish: one resolution is not a mesh-converged
   answer, and status "diverged" means no answer.
6. If the request needs something bessemer cannot do (read_doc
   "capabilities"), say so plainly; never edit the solver to run a case.
"""

DOCS = {
    "using": ("docs/using.md", "How to run cases: decks, Python, results"),
    "capabilities": ("docs/capabilities.md", "What bessemer can and cannot do"),
    "deck_reference": ("docs/deck_reference.md",
                       "Every deck key (generated; deck_keys is always current)"),
    "benchmarks": ("benchmarks/README.md", "The standard benchmarks and their decks"),
    "install": ("docs/install/spack.md", "Installing: scripts/setup.sh and Spack"),
    "assistant": ("docs/assistant.md", "This server: setup and tools"),
}


class ToolError(Exception):
    """An error the assistant should see as the tool's result."""


# --------------------------------------------------------------------------
# Finding the build and running it
# --------------------------------------------------------------------------

_env_ok = None  # cached: does scripts/env.sh activate here?


def find_run_case():
    """The run_case binary, or None."""
    env = os.environ.get("BESSEMER_RUN_CASE")
    if env:
        return env if os.access(env, os.X_OK) else None
    build = os.path.join(ROOT, "build")
    preferred = ["cpu", "cuda", "cpu-python"]
    try:
        presets = preferred + sorted(p for p in os.listdir(build) if p not in preferred)
    except OSError:
        return None
    for p in presets:
        exe = os.path.join(build, p, "apps", "run_case")
        if os.access(exe, os.X_OK):
            return exe
    return None


def use_spack_env():
    """Whether commands run through scripts/env.sh."""
    global _env_ok
    mode = os.environ.get("BESSEMER_ENV", "auto")
    if mode == "none":
        return False
    if mode == "spack":
        return True
    if _env_ok is None:
        script = os.path.join(ROOT, "scripts", "env.sh")
        if not os.path.exists(script):
            _env_ok = False
        else:
            try:
                r = subprocess.run(["bash", "-c", '. "$0" >/dev/null 2>&1', script],
                                   cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=120)
                _env_ok = r.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                _env_ok = False
    return _env_ok


def wrap(argv, mpi_ranks=None):
    """The command line for @p argv: through env.sh when that works, under
    the MPI launcher when @p mpi_ranks is given."""
    if use_spack_env():
        launch = '"${INCNS_MPIEXEC:-mpirun}" -n %d ' % mpi_ranks if mpi_ranks else ""
        script = ('. "%s" >/dev/null 2>&1 || exit 97; exec %s"$@"'
                  % (os.path.join(ROOT, "scripts", "env.sh"), launch))
        return ["bash", "-c", script, "bessemer"] + list(argv)
    if mpi_ranks:
        launcher = shlex.split(os.environ.get("BESSEMER_MPIEXEC", "mpirun"))
        return launcher + ["-n", str(mpi_ranks)] + list(argv)
    return list(argv)


def require_run_case():
    exe = find_run_case()
    if not exe:
        raise ToolError("No run_case binary found under %s/build. Build it with "
                        "`scripts/build.sh cpu` (first `scripts/setup.sh --plan` "
                        "on a new machine), or set BESSEMER_RUN_CASE." % ROOT)
    return exe


def run_quick(argv, timeout=600):
    """Run a short command in the repository; (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(wrap(argv), cwd=ROOT, stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout, universal_newlines=True)
    except subprocess.TimeoutExpired:
        raise ToolError("`%s` took longer than %d s" % (" ".join(argv), timeout))
    return r.returncode, r.stdout, r.stderr


def first_json(text):
    """The first top-level JSON object in @p text, or None."""
    start = text.find("{")
    while start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            return obj
        except ValueError:
            start = text.find("{", start + 1)
    return None


def mfem_error(text):
    """The message of an MFEM error in @p text (MFEM abort / Verification
    failed, up to its "... in function" line), or None."""
    m = re.search(r"(MFEM abort:|Verification failed:)(.*?)(\n \.\.\. in function|\Z)",
                  text, flags=re.S)
    return (m.group(1) + m.group(2)).strip() if m else None


def tail(text, n=30):
    lines = text.rstrip("\n").split("\n")
    return "\n".join(lines[-n:])


def resolve(path):
    """A deck path: absolute, or relative to the repository."""
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        p = os.path.join(ROOT, p)
    return os.path.normpath(p)


def rel(path):
    try:
        r = os.path.relpath(path, ROOT)
    except ValueError:
        return path
    return path if r.startswith("..") else r


def machine_memory():
    """Total memory in bytes, or None."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, universal_newlines=True,
                             timeout=10).stdout
        return int(out.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def gib(b):
    return round(b / 2.0**30, 2)


# --------------------------------------------------------------------------
# Deck text editing (adapted from bench/throughput/bench.py's apply_sets)
# --------------------------------------------------------------------------

def apply_sets(text, sets):
    """The deck text with {("section", "key"): "value"} overrides applied. A key
    the deck already sets has its line replaced (a second line would be a
    duplicate key, which bessemer rejects); a key the deck lacks goes at the end
    of its section, and a section the deck lacks at the end of the deck.
    Block-style maps only: one key per line, nesting by indentation."""
    todo = dict(sets)
    lines = text.rstrip("\n").split("\n")
    stack = []
    maps = {}
    scalars = set()
    for i, line in enumerate(lines):
        body = line.lstrip(" ")
        if not body or body.startswith("#"):
            continue
        indent = len(line) - len(body)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        for d in range(1, len(stack) + 1):
            maps[tuple(k for _, k in stack[:d])]["last"] = i
        if stack:
            parent = maps[tuple(k for _, k in stack)]
            parent["indent"] = parent["indent"] or indent
            parent["seq"] = parent["seq"] or body.startswith("-")
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*):(?:\s+(.*))?$", body)
        if not m:
            continue
        path = tuple(k for _, k in stack) + (m.group(1),)
        if not m.group(2) or m.group(2).startswith("#"):
            if path in todo:
                raise ToolError("set %s: a section in this deck, not a key (use "
                                "new_case's text to rewrite it)" % ".".join(path))
            stack.append((indent, m.group(1)))
            maps[path] = {"indent": 0, "last": i, "seq": False, "at": indent}
        else:
            scalars.add(path)
            if path in todo:
                lines[i] = " " * indent + "%s: %s" % (m.group(1), todo.pop(path))
    inserts = {}
    for path, value in todo.items():
        d = len(path) - 1
        while d and path[:d] not in maps:
            d -= 1
        base = path[:d]
        if path[:d + 1] in scalars:
            raise ToolError("set %s: %s is a key, not a section"
                            % (".".join(path), ".".join(path[:d + 1])))
        if base and maps[base]["seq"]:
            raise ToolError("set %s: %s is a list (use new_case's text to rewrite "
                            "it)" % (".".join(path), ".".join(base)))
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
    for base in sorted((b for b in inserts if b),
                       key=lambda b: (-maps[b]["last"], len(b))):
        s = maps[base]
        lines[s["last"] + 1:s["last"] + 1] = render(inserts[base],
                                                    s["indent"] or s["at"] + 2)
    lines += render(inserts.get((), {}), 0)
    return "\n".join(lines) + "\n"


def yaml_scalar(v):
    """A JSON value as YAML flow text (YAML is a superset of JSON)."""
    if isinstance(v, str):
        return v  # given as YAML text already: 1e-3, [1, 0], navier_stokes
    return json.dumps(v)


def parse_sets(sets):
    out = {}
    for key, value in (sets or {}).items():
        path = tuple(key.split("."))
        if not all(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k) for k in path):
            raise ToolError("set: %r is not a dotted deck key such as physics.nu" % key)
        out[path] = yaml_scalar(value)
    for p in out:
        for q in out:
            if len(q) > len(p) and q[:len(p)] == p:
                raise ToolError("set: %s conflicts with %s" % (".".join(p), ".".join(q)))
    return out


def header_comment(text, max_lines=12):
    out = []
    for line in text.split("\n"):
        if line.startswith("#"):
            out.append(line.lstrip("#").strip())
        elif line.strip() == "" and not out:
            continue
        else:
            break
    while out and not out[-1]:
        out.pop()
    return "\n".join(out[:max_lines])


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

def tool_check_setup(args):
    exe = find_run_case()
    out = {
        "repository": ROOT,
        "run_case": exe,
        "cores": os.cpu_count(),
        "memory_gib": gib(machine_memory()) if machine_memory() else None,
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           universal_newlines=True, timeout=10)
        out["commit"] = r.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        out["commit"] = None
    out["toolchain"] = "spack (scripts/env.sh)" if use_spack_env() else "as on PATH"
    if not use_spack_env():
        launcher = shlex.split(os.environ.get("BESSEMER_MPIEXEC", "mpirun"))[0]
        out["mpi_launcher"] = shutil.which(launcher)
    if exe:
        rc, so, se = run_quick([exe, "--deck-schema"], timeout=120)
        schema = first_json(so) if rc == 0 else None
        out["deck_keys"] = len(schema) if schema else None
        if schema is None:
            out["problem"] = ("run_case did not print its deck schema (an older "
                              "build? rebuild with scripts/build.sh cpu): "
                              + tail(se, 10))
        else:
            out["ready"] = True
    else:
        out["ready"] = False
        out["next_step"] = ("No build found. Run `scripts/setup.sh --plan` and "
                            "show the user the plan (what gets installed where). "
                            "Run `scripts/setup.sh --yes` only after they agree; "
                            "then `scripts/build.sh cpu`. read_doc install has "
                            "the details.")
    return out


def tool_list_examples(args):
    query = (args.get("query") or "").lower()
    out = []
    paths = []
    for d in ["cases"] + ["benchmarks/" + b for b in sorted(os.listdir(os.path.join(ROOT, "benchmarks")))
                          if os.path.isdir(os.path.join(ROOT, "benchmarks", b))]:
        full = os.path.join(ROOT, d)
        if os.path.isdir(full):
            paths += [os.path.join(d, f) for f in sorted(os.listdir(full)) if f.endswith(".yaml")]
    for p in paths:
        try:
            text = open(os.path.join(ROOT, p)).read()
        except OSError:
            continue
        m = re.search(r"^equation:\s*(\S+)", text, flags=re.M)
        g = re.search(r"^\s+geometry:\s*(\S+)", text, flags=re.M)
        d = re.search(r"^\s+dim:\s*(\d)", text, flags=re.M)
        entry = {"path": p,
                 "equation": m.group(1) if m else "stokes (default)",
                 "geometry": g.group(1) if g else "box",
                 "dim": int(d.group(1)) if d else None,
                 "about": header_comment(text, 8)}
        if query and query not in (p + " " + entry["about"] + " " + entry["equation"]).lower():
            continue
        out.append(entry)
    return {"examples": out,
            "note": "Start from the closest one with new_case. benchmarks/<case>/README.md "
                    "has results to compare against (read_doc benchmarks)."}


_schema_cache = {}


def deck_schema():
    exe = require_run_case()
    key = (exe, os.path.getmtime(exe))
    if key not in _schema_cache:
        rc, so, se = run_quick([exe, "--deck-schema"], timeout=120)
        schema = first_json(so) if rc == 0 else None
        if schema is None:
            raise ToolError("run_case --deck-schema failed (rebuild with "
                            "scripts/build.sh cpu?):\n" + tail(se, 20))
        _schema_cache.clear()
        _schema_cache[key] = schema
    return _schema_cache[key]


def tool_deck_keys(args):
    schema = deck_schema()
    query = (args.get("query") or "").lower()
    section = args.get("section") or ""
    limit = int(args.get("limit") or 60)
    hits = []
    for path, k in schema.items():
        if section and not (path == section or path.startswith(section + ".")):
            continue
        if query and query not in path.lower() and query not in k.get("description", "").lower():
            continue
        hits.append(dict(path=path, **k))
    sections = sorted({p.split(".")[0] for p in schema if "." in p})
    return {"matches": len(hits), "keys": hits[:limit],
            "truncated": len(hits) > limit, "sections": sections,
            "source": "this machine's run_case (--deck-schema): the defaults of this build"}


def tool_read_doc(args):
    name = args.get("name") or ""
    if name not in DOCS:
        bench = os.path.join(ROOT, "benchmarks", name, "README.md")
        if os.path.exists(bench):
            return {"name": name, "path": rel(bench), "text": open(bench).read()}
        raise ToolError("unknown doc %r; one of %s or a benchmark directory name "
                        "(list_examples)" % (name, ", ".join(sorted(DOCS))))
    path = os.path.join(ROOT, DOCS[name][0])
    if not os.path.exists(path):
        raise ToolError("%s is not in this checkout" % DOCS[name][0])
    return {"name": name, "path": DOCS[name][0], "text": open(path).read()}


def check(deck):
    """run_case DECK --check, as a dict."""
    exe = require_run_case()
    if not os.path.exists(deck):
        raise ToolError("no deck at %s" % deck)
    rc, so, se = run_quick([exe, deck, "--check"])
    report = first_json(so)
    if report is None:
        # An MFEM error aborts before the report (MFEM built without
        # exceptions): its message is on stderr.
        return {"status": "error", "deck": rel(deck), "error": mfem_error(se + so)
                or tail(se or so, 25) or "run_case --check exited %d" % rc}
    report["deck"] = rel(deck)
    if report.get("status") == "ok":
        dofs = report.get("dofs", {})
        n = dofs.get("velocity") or dofs.get("displacement")
        if n:
            lo, hi = (n * b for b in BYTES_PER_VDOF)
            est = {"memory_gib": [gib(lo), gib(hi)],
                   "basis": "0.6-1.5 kB per velocity DOF, measured on GPUs "
                            "(docs/gpu_memory.md); host memory for a CPU run is "
                            "similar. AMR grows it during the run."}
            mem = machine_memory()
            if mem:
                est["machine_memory_gib"] = gib(mem)
                if hi > 0.8 * mem:
                    est["warning"] = ("may not fit in this machine's memory: "
                                      "coarsen the mesh or lower the order")
            report["estimate"] = est
        t = report.get("time", {})
        if "steps" not in t and report.get("equation") not in (
                "linear_elasticity", "hyperelasticity", "mesh_motion"):
            t["steps_note"] = ("dt adapts (CFL or error control), so the step count "
                               "is known only once it runs: run_status gives an ETA "
                               "from the first progress lines")
    return report


def tool_check_deck(args):
    return check(resolve(args["path"]))


def tool_new_case(args):
    name = args.get("name") or ""
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$", name):
        raise ToolError("name: letters, digits, '_' and '-' only (it names the "
                        "directory runs/<name>/ and the output files)")
    template, text = args.get("template"), args.get("text")
    if bool(template) == bool(text):
        raise ToolError("give exactly one of template (a deck to start from, e.g. "
                        "cases/cavity.yaml) or text (a whole deck)")
    if template:
        src = resolve(template)
        if not os.path.exists(src):
            raise ToolError("no deck at %s (list_examples shows them)" % template)
        text = open(src).read()
        text = ("# runs/%s/%s.yaml: from %s (new_case)\n" % (name, name, rel(src))) + text
    sets = parse_sets(args.get("set"))
    outdir = os.path.join(RUNS, name)
    sets.setdefault(("output", "path"), rel(outdir))
    sets.setdefault(("output", "name"), name)
    text = apply_sets(text, sets)
    deck = os.path.join(outdir, name + ".yaml")
    if os.path.exists(deck) and not args.get("overwrite"):
        raise ToolError("%s exists; pass overwrite: true to replace it" % rel(deck))
    os.makedirs(outdir, exist_ok=True)
    with open(deck, "w") as f:
        f.write(text)
    report = check(deck)
    # A run that reports its progress, so that run_status has an ETA.
    if report.get("status") == "ok" and not re.search(r"^\s+progress:", text, flags=re.M) \
            and report.get("equation") in ("stokes", "navier_stokes"):
        t_final = report.get("time", {}).get("t_final")
        if t_final:
            text = apply_sets(text, {("output", "progress"): repr(t_final / 50.0)})
            with open(deck, "w") as f:
                f.write(text)
            report["added"] = "output.progress = t_final/50, for run_status's ETA"
    return {"deck": rel(deck), "check": report, "text": text}


def tool_write_deck(args):
    """Overwrite a deck under runs/ with new text, then check it."""
    deck = resolve(args["path"])
    if not deck.startswith(RUNS + os.sep):
        raise ToolError("write_deck edits decks under runs/ only; copy an example "
                        "there with new_case first")
    os.makedirs(os.path.dirname(deck), exist_ok=True)
    with open(deck, "w") as f:
        f.write(args["text"] if args["text"].endswith("\n") else args["text"] + "\n")
    return {"deck": rel(deck), "check": check(deck)}


def run_dir(run_id):
    if not re.match(r"^[A-Za-z0-9_\-]+$", run_id or ""):
        raise ToolError("bad run id %r" % run_id)
    d = os.path.join(REGISTRY, run_id)
    if not os.path.exists(os.path.join(d, "run.json")):
        raise ToolError("no run %r (run_status without an id lists them)" % run_id)
    return d


def load_run(run_id):
    d = run_dir(run_id)
    with open(os.path.join(d, "run.json")) as f:
        meta = json.load(f)
    meta["_dir"] = d
    return meta


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie child of this server counts as finished.
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


def tool_run_deck(args):
    deck = resolve(args["path"])
    report = check(deck)
    if report.get("status") != "ok":
        return {"started": False, "check": report,
                "note": "fix the deck first (check_deck reports the same)"}
    cores = os.cpu_count() or 1
    ranks = int(args.get("ranks") or min(4, cores))
    if ranks < 1 or ranks > cores:
        raise ToolError("ranks must be between 1 and this machine's %d cores" % cores)
    exe = require_run_case()
    os.makedirs(REGISTRY, exist_ok=True)
    stem = os.path.splitext(os.path.basename(deck))[0]
    run_id = "%s-%s" % (stem, time.strftime("%Y%m%d-%H%M%S"))
    d = os.path.join(REGISTRY, run_id)
    os.makedirs(d)
    log = os.path.join(d, "log.txt")
    cmd = wrap([exe, deck], mpi_ranks=ranks)
    # The exit code survives this server: a shell writes it when the run ends.
    shell = 'cd "$1" && shift && "$@"; echo $? > "%s"' % os.path.join(d, "exit_code")
    with open(log, "w") as f:
        proc = subprocess.Popen(["sh", "-c", shell, "sh", ROOT] + cmd,
                                stdin=subprocess.DEVNULL, stdout=f,
                                stderr=subprocess.STDOUT, start_new_session=True)
    summary = os.path.join(ROOT, report.get("output_summary", ""))
    meta = {"id": run_id, "deck": rel(deck), "ranks": ranks, "pid": proc.pid,
            "started": time.time(), "log": rel(log), "summary": rel(summary),
            "t_final": report.get("time", {}).get("t_final"),
            "command": " ".join(shlex.quote(c) for c in cmd)}
    with open(os.path.join(d, "run.json"), "w") as f:
        json.dump(meta, f, indent=2)
    out = {"started": True, "id": run_id, "ranks": ranks, "log": meta["log"],
           "summary": meta["summary"], "check": report}
    if not summary.startswith(RUNS + os.sep):
        out["note"] = ("this deck writes its output outside runs/ (output.path): "
                       "user runs belong under runs/ (new_case sets that up)")
    return out


PROGRESS = re.compile(r"^t\s+(\S+)\s+dt\s+(\S+).*?CFL\s+(\S+).*?its\s+(\d+).*?"
                      r"elements\s+(\d+)\s+steps\s+(\d+)\s+wall\s+(\S+)\s+s\s+ETA\s+(\S+)\s+min")


def run_state(meta):
    d = meta["_dir"]
    code_file = os.path.join(d, "exit_code")
    alive = pid_alive(meta["pid"])
    state = {"id": meta["id"], "deck": meta["deck"], "ranks": meta["ranks"],
             "elapsed_s": round(time.time() - meta["started"], 1)}
    if os.path.exists(code_file):
        try:
            code = int(open(code_file).read().strip() or "-1")
        except ValueError:
            code = -1
        state["state"] = {0: "finished", 2: "diverged"}.get(code, "failed")
        state["exit_code"] = code
        state["elapsed_s"] = round(os.path.getmtime(code_file) - meta["started"], 1)
    elif alive:
        state["state"] = "running"
    else:
        state["state"] = "stopped"
    return state


def tool_run_status(args):
    run_id = args.get("id")
    if not run_id:
        runs = []
        if os.path.isdir(REGISTRY):
            for r in sorted(os.listdir(REGISTRY), reverse=True):
                try:
                    runs.append(run_state(load_run(r)))
                except (ToolError, OSError, ValueError):
                    pass
        return {"runs": runs[:30]}
    meta = load_run(run_id)
    state = run_state(meta)
    log = os.path.join(ROOT, meta["log"])
    text = open(log, errors="replace").read() if os.path.exists(log) else ""
    progress = [l for l in text.split("\n") if PROGRESS.match(l)]
    if progress:
        m = PROGRESS.match(progress[-1])
        state["progress"] = {"t": float(m.group(1)), "t_final": meta.get("t_final"),
                             "dt": float(m.group(2)), "cfl": float(m.group(3)),
                             "iterations": int(m.group(4)), "elements": int(m.group(5)),
                             "steps": int(m.group(6)), "wall_s": float(m.group(7)),
                             "eta_min": float(m.group(8))}
        state["progress_lines"] = progress[-5:]
    errors = [l for l in text.split("\n")
              if re.search(r"MFEM abort|Verification failed|terminate called|"
                           r"Error|error:|Segmentation|Killed", l)]
    if errors:
        state["errors"] = errors[-8:]
    state["log_tail"] = tail(text, int(args.get("lines") or 15))
    summary = os.path.join(ROOT, meta["summary"])
    if os.path.exists(summary):
        try:
            state["summary_status"] = json.load(open(summary)).get("status")
        except ValueError:
            pass
    if state["state"] == "running" and not progress:
        state["note"] = ("no progress line yet (the first comes after setup and "
                         "output.progress time units)")
    return state


def tool_stop_run(args):
    meta = load_run(args["id"])
    if not pid_alive(meta["pid"]):
        return {"id": meta["id"], "stopped": False, "note": "not running"}
    try:
        os.killpg(meta["pid"], signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(50):
        if not pid_alive(meta["pid"]):
            break
        time.sleep(0.2)
    else:
        try:
            os.killpg(meta["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
    with open(os.path.join(meta["_dir"], "exit_code"), "w") as f:
        f.write("-15\n")
    return {"id": meta["id"], "stopped": True,
            "note": "the summary keeps status 'running' for a stopped run; if MPI "
                    "processes linger, scripts/reap.sh lists and ends them"}


def numeric_columns(rows, header):
    stats = {}
    for j, name in enumerate(header):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[j]))
            except (ValueError, IndexError):
                pass
        if vals:
            stats[name] = {"first": vals[0], "last": vals[-1],
                           "min": min(vals), "max": max(vals)}
    return stats


def tool_read_results(args):
    if args.get("id"):
        meta = load_run(args["id"])
        summary = os.path.join(ROOT, meta["summary"])
    elif args.get("summary"):
        summary = resolve(args["summary"])
    else:
        raise ToolError("give a run id (run_status lists them) or a summary path")
    if not os.path.exists(summary):
        raise ToolError("no summary at %s yet (the run writes it when it starts)" % rel(summary))
    data = json.load(open(summary))
    profile = data.pop("profile", None)
    out = {"summary_path": rel(summary), "summary": data}
    if profile and isinstance(profile.get("regions"), dict):
        top = sorted(((k, v.get("seconds", 0)) for k, v in profile["regions"].items()
                      if k.count("/") <= 1), key=lambda kv: -kv[1])[:8]
        out["profile_top"] = {k: round(s, 3) for k, s in top}
    outdir = os.path.dirname(summary)
    stem = os.path.basename(summary)[:-len("_summary.json")] \
        if summary.endswith("_summary.json") else None
    files = sorted(f for f in os.listdir(outdir)
                   if stem is None or f.startswith(stem)) if os.path.isdir(outdir) else []
    out["files"] = [rel(os.path.join(outdir, f)) for f in files][:50]
    if stem:
        hist = os.path.join(outdir, stem + "_history.csv")
        if os.path.exists(hist):
            lines = [l for l in open(hist).read().split("\n") if l and not l.startswith("#")]
            if lines:
                header = lines[0].split(",")
                rows = [l.split(",") for l in lines[1:]]
                out["history"] = {"path": rel(hist), "rows": len(rows),
                                  "columns": numeric_columns(rows, header)}
    status = data.get("status")
    notes = []
    if status == "diverged":
        notes.append("status diverged: this run gives no answer; smaller steps "
                     "(time.cfl_target or time.dt) or a finer mesh are the usual cures")
    elif status == "running":
        notes.append("status running: the run is still going, or it died (run_status "
                     "tells which); the values are not final")
    if "reference" in data:
        notes.append("'reference' compares with the deck's reference values")
    notes.append("one resolution is not a mesh-converged answer: compare a finer mesh "
                 "or a higher order before quoting digits")
    out["notes"] = notes
    return out


TOOLS = [
    ("check_setup", tool_check_setup,
     "Check this machine's bessemer setup: the run_case build, cores, memory, the "
     "MPI toolchain. Call first. If nothing is built it says how to set up.",
     {"type": "object", "properties": {}}),
    ("list_examples", tool_list_examples,
     "The example and benchmark decks (cases/, benchmarks/) with their header "
     "comments, to pick the closest starting point for a new case.",
     {"type": "object", "properties": {
         "query": {"type": "string", "description": "Filter, e.g. cylinder, channel, 3d"}}}),
    ("deck_keys", tool_deck_keys,
     "Search every deck key (type, default, allowed values, description), read "
     "from this machine's run_case so it matches the build. Use it instead of "
     "guessing a key or a default.",
     {"type": "object", "properties": {
         "query": {"type": "string", "description": "Substring of a key or its description"},
         "section": {"type": "string", "description": "A section, e.g. time, solver, mesh.cylinder_channel"},
         "limit": {"type": "integer", "description": "Max keys returned (default 60)"}}}),
    ("read_doc", tool_read_doc,
     "Read a document: " + "; ".join("%s (%s)" % (k, v[1]) for k, v in DOCS.items())
     + "; or a benchmark directory name for its README with results.",
     {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}),
    ("new_case", tool_new_case,
     "Create runs/<name>/<name>.yaml from an example deck (template, keeping its "
     "comments) with dotted-key changes (set, e.g. {\"physics.nu\": 1e-3, "
     "\"mesh.elements\": \"[32, 32]\"}), or from a whole deck (text). Output goes to "
     "runs/<name>/. Then checks it (as check_deck). Lists such as "
     "boundary_conditions are changed through text or write_deck.",
     {"type": "object", "properties": {
         "name": {"type": "string"},
         "template": {"type": "string", "description": "e.g. cases/cavity.yaml"},
         "set": {"type": "object", "description": "dotted key -> value (YAML text or JSON value)"},
         "text": {"type": "string", "description": "A whole deck instead of a template"},
         "overwrite": {"type": "boolean"}},
      "required": ["name"]}),
    ("write_deck", tool_write_deck,
     "Replace a deck under runs/ with new text (YAML), then check it.",
     {"type": "object", "properties": {
         "path": {"type": "string"}, "text": {"type": "string"}},
      "required": ["path", "text"]}),
    ("check_deck", tool_check_deck,
     "Vet a deck without running it: load it (unknown keys, bad values), build the "
     "mesh, check every boundary is in exactly one group, and count the DOFs. "
     "Returns the loader's error, or the size, the settings in effect (convection "
     "scheme, BDF order, CFL target...), a memory estimate and the step count "
     "when dt is fixed.",
     {"type": "object", "properties": {
         "path": {"type": "string", "description": "Deck path, absolute or relative to the repository"}},
      "required": ["path"]}),
    ("run_deck", tool_run_deck,
     "Check a deck, then run it in the background on this machine (MPI, ranks "
     "default min(4, cores)). Returns a run id for run_status and read_results. "
     "Say the expected cost to the user before long runs.",
     {"type": "object", "properties": {
         "path": {"type": "string"},
         "ranks": {"type": "integer", "description": "MPI ranks, at most the core count"}},
      "required": ["path"]}),
    ("run_status", tool_run_status,
     "A run's state (running, finished, diverged, failed, stopped), its latest "
     "progress (t, dt, CFL, iterations, wall time, ETA from the run itself), "
     "errors and the log's tail. Without an id: the recent runs.",
     {"type": "object", "properties": {
         "id": {"type": "string"}, "lines": {"type": "integer"}}}),
    ("stop_run", tool_stop_run,
     "Stop a run started by run_deck (its whole process group).",
     {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}),
    ("read_results", tool_read_results,
     "A run's results from its summary.json (status, time, mesh, forces, "
     "reference errors, diagnostics...), the history CSV's ranges, the output "
     "files, and what the run does not establish. By run id or summary path.",
     {"type": "object", "properties": {
         "id": {"type": "string"}, "summary": {"type": "string"}}}),
]
TOOL_FUNCS = {name: fn for name, fn, _, _ in TOOLS}


# --------------------------------------------------------------------------
# MCP over stdio
# --------------------------------------------------------------------------

def resources():
    out = []
    for name, (path, desc) in DOCS.items():
        if os.path.exists(os.path.join(ROOT, path)):
            out.append({"uri": "bessemer://docs/" + name, "name": name,
                        "description": desc, "mimeType": "text/markdown"})
    return out


def handle(msg):
    """The response to one JSON-RPC message (None for a notification)."""
    method = msg.get("method")
    mid = msg.get("id")
    params = msg.get("params") or {}
    if mid is None:
        return None  # notifications: initialized, cancelled, ...

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def err(code, message):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}

    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return ok({"protocolVersion": version,
                   "capabilities": {"tools": {}, "resources": {}},
                   "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                   "instructions": INSTRUCTIONS})
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": [{"name": n, "description": d, "inputSchema": s}
                             for n, _, d, s in TOOLS]})
    if method == "tools/call":
        name = params.get("name")
        if name not in TOOL_FUNCS:
            return err(-32602, "unknown tool %r" % name)
        try:
            result = TOOL_FUNCS[name](params.get("arguments") or {})
            return ok({"content": [{"type": "text", "text": json.dumps(result, indent=1)}],
                       "structuredContent": result, "isError": False})
        except ToolError as e:
            return ok({"content": [{"type": "text", "text": str(e)}], "isError": True})
        except KeyError as e:
            return ok({"content": [{"type": "text", "text": "missing argument %s" % e}],
                       "isError": True})
        except Exception as e:  # report, keep serving
            return ok({"content": [{"type": "text", "text": "%s: %s" % (type(e).__name__, e)}],
                       "isError": True})
    if method == "resources/list":
        return ok({"resources": resources()})
    if method == "resources/read":
        uri = params.get("uri", "")
        name = uri.rsplit("/", 1)[-1]
        if not uri.startswith("bessemer://docs/") or name not in DOCS:
            return err(-32002, "unknown resource %r" % uri)
        text = open(os.path.join(ROOT, DOCS[name][0])).read()
        return ok({"contents": [{"uri": uri, "mimeType": "text/markdown", "text": text}]})
    if method in ("prompts/list",):
        return ok({"prompts": []})
    return err(-32601, "method not found: %s" % method)


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None,
                     "error": {"code": -32700, "message": "parse error"}}
        else:
            batch = msg if isinstance(msg, list) else [msg]
            replies = [r for r in (handle(m) for m in batch) if r is not None]
            reply = replies if isinstance(msg, list) else (replies[0] if replies else None)
        if reply:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
