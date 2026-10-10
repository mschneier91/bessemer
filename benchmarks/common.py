"""Shared helpers for the benchmark plot scripts.

Standard library only, except plotting: matplotlib is imported lazily, and a
script without it still prints its comparison table (the desktop Spack
environment gains matplotlib at its next refresh; environments/stack.yaml
already lists it).
"""

import json
import os
import sys

# Exit code of a script whose reference data has not been fetched; the
# fast-tier self-tests report it as skipped (SKIP_RETURN_CODE).
SKIP = 77


def read_csv(path):
    """Columns of a CSV whose header is the first line not starting with '#'.

    Returns (columns: dict name -> list of floats, comments: dict). Comment
    lines of the form '# key value' become comments[key] = value (a float
    when it parses as one).
    """
    cols, names, comments = {}, None, {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                parts = s[1:].split(None, 1)
                if len(parts) == 2:
                    try:
                        comments[parts[0]] = float(parts[1])
                    except ValueError:
                        comments[parts[0]] = parts[1]
                continue
            if names is None:
                names = [n.strip() for n in s.split(",")]
                cols = {n: [] for n in names}
                continue
            for n, v in zip(names, s.split(",")):
                cols[n].append(float(v))
    if names is None:
        raise ValueError(path + ": no header line")
    return cols, comments


def last_per_time(cols, key="t"):
    """Sort by time, keeping the last row for a repeated time (a restarted
    run appends rows the interrupted run had already written)."""
    seen = {}
    for i, t in enumerate(cols[key]):
        seen[t] = i
    order = [seen[t] for t in sorted(seen)]
    return {n: [v[i] for i in order] for n, v in cols.items()}


def interp(x, xs, ys):
    """Linear interpolation of (xs, ys) at x; xs increasing; None outside."""
    if not xs or x < xs[0] or x > xs[-1]:
        return None
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    if xs[hi] == xs[lo]:
        return ys[lo]
    w = (x - xs[lo]) / (xs[hi] - xs[lo])
    return (1 - w) * ys[lo] + w * ys[hi]


def peak(ts, vs, find_min=False):
    """(value, time) of the extreme of a sampled signal, refined by the
    parabola through the extreme sample and its neighbours."""
    sign = -1.0 if find_min else 1.0
    i = max(range(len(vs)), key=lambda k: sign * vs[k])
    if 0 < i < len(vs) - 1:
        t0, t1, t2 = ts[i - 1], ts[i], ts[i + 1]
        v0, v1, v2 = vs[i - 1], vs[i], vs[i + 1]
        den = (t0 - t1) * (t0 - t2) * (t1 - t2)
        if den != 0.0:
            a = (t2 * (v1 - v0) + t1 * (v0 - v2) + t0 * (v2 - v1)) / den
            b = (t2 * t2 * (v0 - v1) + t1 * t1 * (v2 - v0)
                 + t0 * t0 * (v1 - v2)) / den
            if a * sign < 0.0:
                tp = -b / (2 * a)
                if t0 <= tp <= t2:
                    c = v1 - a * t1 * t1 - b * t1
                    return a * tp * tp + b * tp + c, tp
    return vs[i], ts[i]


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def pyplot():
    """matplotlib.pyplot (Agg backend), or None with a note."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("(matplotlib not available: comparison table only; it is in "
              "environments/stack.yaml)")
        return None


def reference_path(script, *parts):
    """A path inside the benchmark directory of @p script (its __file__)."""
    return os.path.join(os.path.dirname(os.path.abspath(script)), *parts)


def read_reference(script, *parts):
    """A benchmark's reference CSV (read_csv). The data is not committed (its
    sources state no redistribution terms): when it has not been fetched yet,
    say how and exit with SKIP."""
    path = reference_path(script, *parts)
    if not os.path.exists(path):
        print("reference data not fetched: %s" % os.path.relpath(path))
        print("  fetch it once: python3 benchmarks/fetch_references.py")
        print("  (not committed: the sources state no redistribution terms)")
        sys.exit(SKIP)
    return read_csv(path)
