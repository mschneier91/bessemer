#!/usr/bin/env python3
"""Turbulent channel: compare a run's profiles with Moser, Kim & Mansour.

    python3 benchmarks/channel/plot.py RUN_PROFILES.csv [--re 180|395] [-o out.png]
    python3 benchmarks/channel/plot.py --selftest

RUN_PROFILES.csv is the run's <path>/<name>_profiles.csv (channel_statistics):
time and x-z plane averages at every node height, with u_tau and Re_tau from
the averaged wall shear stress in its header. The two halves are folded onto
the wall distance (u'v' changes sign in the upper half) and compared in wall
units with the DNS at the nearest Re_tau (or --re).

Printed: Re_tau (run vs DNS), the bulk velocity, and the largest differences
in U+ (relative, y+ >= 5) and in the rms velocities and u'v' (wall units)
over the run's stations. With matplotlib, also U+(y+) with the law of the
wall, and the Reynolds stresses.
"""

import argparse
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common  # noqa: E402

DNS = {"180": ("mkm_chan180.csv", 178.12), "395": ("mkm_chan395.csv", 392.24)}


def reference(key):
    cols, _ = common.read_reference(__file__, "reference", DNS[key][0])
    return cols


def fold(cols):
    """Average the two halves at equal wall distance (stations of a
    symmetric mesh pair up exactly; others stay single)."""
    y = cols["y"]
    y0, y1 = y[0], y[-1]
    groups = {}
    for i, yy in enumerate(y):
        upper = (yy - y0) > (y1 - yy)
        d = (y1 - yy) if upper else (yy - y0)
        key = round(d / (y1 - y0), 9)
        sign = -1.0 if upper else 1.0
        groups.setdefault(key, []).append((i, sign))
    out = {n: [] for n in ("d", "y_plus", "U_plus", "urms_plus", "vrms_plus",
                           "wrms_plus", "uv_plus")}
    for key in sorted(groups):
        members = groups[key]
        out["d"].append(key * (y1 - y0))
        for n in ("y_plus", "U_plus", "urms_plus", "vrms_plus", "wrms_plus"):
            out[n].append(sum(cols[n][i] for i, _ in members) / len(members))
        out["uv_plus"].append(sum(s * cols["uv_plus"][i] for i, s in members)
                              / len(members))
    return out


def compare(run, ref):
    err = {"U_plus": 0.0, "urms_plus": 0.0, "vrms_plus": 0.0, "wrms_plus": 0.0,
           "uv_plus": 0.0}
    for k, yp in enumerate(run["y_plus"]):
        for n in err:
            r = common.interp(yp, ref["y_plus"], ref[n])
            if r is None:
                continue
            if n == "U_plus":
                if yp >= 5.0:
                    err[n] = max(err[n], abs(run[n][k] - r) / abs(r))
            else:
                err[n] = max(err[n], abs(run[n][k] - r))
    return err


def report(name, comments, ref_key, err):
    print("%s: Re_tau %.2f (DNS %.2f), U_bulk %s, averaging time %s (%s samples)"
          % (name, comments.get("re_tau", float("nan")), DNS[ref_key][1],
             comments.get("u_bulk", "?"), comments.get("averaging_time", "?"),
             comments.get("samples", "?")))
    print("  max |U+ - DNS| / DNS (y+ >= 5) %.2f%%; max abs differences in wall "
          "units: urms %.3f, vrms %.3f, wrms %.3f, u'v' %.3f"
          % (100 * err["U_plus"], err["urms_plus"], err["vrms_plus"],
             err["wrms_plus"], err["uv_plus"]))


def plot(name, run, ref, ref_key, out):
    plt = common.pyplot()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    yp = [v for v in run["y_plus"] if v > 0]
    ax[0].semilogx(ref["y_plus"][1:], ref["U_plus"][1:], "k-", lw=2,
                   label="MKM Re_tau %.0f" % DNS[ref_key][1])
    ax[0].semilogx(yp, [u for v, u in zip(run["y_plus"], run["U_plus"]) if v > 0],
                   "o", ms=3, label=name)
    lin = [v for v in ref["y_plus"][1:] if v < 12]
    ax[0].semilogx(lin, lin, ":", color="gray", label="U+ = y+")
    log = [v for v in ref["y_plus"][1:] if v > 20]
    ax[0].semilogx(log, [2.5 * math.log(v) + 5.5 for v in log], "--", color="gray",
                   label="2.5 ln y+ + 5.5")
    ax[0].set(xlabel="y+", ylabel="U+", title="Mean velocity")
    for n, c in (("urms_plus", "C0"), ("vrms_plus", "C1"), ("wrms_plus", "C2"),
                 ("uv_plus", "C3")):
        ax[1].plot(ref["y_plus"], ref[n], "-", color=c, label=n + " (DNS)")
        ax[1].plot(run["y_plus"], run[n], "o", color=c, ms=3)
    ax[1].set(xlabel="y+", title="Reynolds stresses (wall units; dots: run)")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


def write_fake_run(path, ref, re_tau):
    """The DNS as a run's profiles CSV: both halves, u_tau = delta = 1."""
    nu = 1.0 / re_tau
    rows = []
    for i in range(len(ref["y"])):
        rows.append((ref["y"][i], i, 1.0))
    for i in reversed(range(len(ref["y"]) - 1)):
        rows.append((2.0 - ref["y"][i], i, -1.0))
    with open(path, "w") as f:
        f.write("# nu %.17g\n# delta 1\n# u_tau 1\n# re_tau %.17g\n# u_bulk 0\n"
                % (nu, re_tau))
        f.write("y,y_plus,U,V,W,uu,vv,ww,uv,U_plus,urms_plus,vrms_plus,wrms_plus,"
                "uv_plus\n")
        for y, i, s in rows:
            u, ur, vr, wr, uv = (ref["U_plus"][i], ref["urms_plus"][i],
                                 ref["vrms_plus"][i], ref["wrms_plus"][i],
                                 s * ref["uv_plus"][i])
            f.write(",".join("%.17g" % v for v in
                             (y, ref["y_plus"][i], u, 0, 0, ur * ur, vr * vr,
                              wr * wr, uv, u, ur, vr, wr, uv)) + "\n")


def selftest():
    status = 0
    for key in DNS:
        ref = reference(key)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fake_profiles.csv")
            write_fake_run(path, ref, DNS[key][1])
            cols, comments = common.read_csv(path)
            run = fold(cols)
            err = compare(run, ref)
        report("selftest Re_tau %s (DNS as run)" % key, comments, key, err)
        ok = len(run["y_plus"]) == len(ref["y_plus"]) and max(err.values()) < 1e-12
        print("selftest", key, "ok" if ok else "FAILED")
        status |= 0 if ok else 1
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("profiles", nargs="?", help="<name>_profiles.csv")
    ap.add_argument("--re", choices=sorted(DNS), help="DNS to compare with")
    ap.add_argument("-o", "--out", default="channel.png")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.profiles:
        ap.error("give a profiles CSV (or --selftest)")
    cols, comments = common.read_csv(a.profiles)
    if "U_plus" not in cols:
        ap.error(a.profiles + " has no samples yet")
    key = a.re or min(DNS, key=lambda k: abs(DNS[k][1] - comments.get("re_tau", 0)))
    ref = reference(key)
    run = fold(cols)
    name = os.path.basename(a.profiles)
    report(name, comments, key, compare(run, ref))
    plot(name, run, ref, key, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
