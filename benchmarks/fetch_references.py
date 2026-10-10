#!/usr/bin/env python3
"""Download the benchmarks' reference data from the original sources and write
them as CSV files, each with its citation in a commented header.

    python3 benchmarks/fetch_references.py          # all of them
    python3 benchmarks/fetch_references.py channel  # one benchmark

The CSVs are not committed (the sources state no redistribution terms; they
are git-ignored): run this once per checkout before comparing runs. The plot
scripts say so when the data is missing. Standard library only.
"""

import math
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("latin-1")


def rows(text):
    """Numeric rows of a whitespace-separated table, comments skipped."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("%"):
            continue
        try:
            out.append([float(v.replace("D", "E")) for v in s.split()])
        except ValueError:
            continue  # a column-title line
    return out


def write(path, header, columns, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for line in header:
            f.write("# " + line + "\n")
        f.write(",".join(columns) + "\n")
        for r in data:
            f.write(",".join("%.10g" % v for v in r) + "\n")
    print("wrote", os.path.relpath(path, os.getcwd()), "(%d rows)" % len(data))


def tgv():
    url = "https://cfd.ku.edu/hiocfd/spectral_Re1600_512.gdiag"
    data = rows(fetch(url))
    write(os.path.join(HERE, "tgv_re1600", "reference", "spectral_re1600_512.csv"),
          ["Taylor-Green vortex, Re = 1600: dealiased pseudo-spectral DNS on 512^3",
           "(reference of the International Workshops on High-Order CFD Methods,",
           "case C3.5; Wang et al., Int. J. Numer. Meth. Fluids 72 (2013) 811-845).",
           "Source: " + url,
           "Nondimensional: V0 = L = 1 on [-pi L, pi L]^3, t in L / V0.",
           "E = (1/|Omega|) int |u|^2 / 2; eps = -dE/dt; enstrophy =",
           "(1/|Omega|) int |omega|^2 / 2 (eps = 2 nu enstrophy for this flow)."],
          ["t", "E", "eps", "enstrophy"], data)


def channel():
    base = "https://turbulence.oden.utexas.edu/data/MKM/"
    for name in ("chan180", "chan395"):
        means = rows(fetch(base + name + "/profiles/" + name + ".means"))
        rey = rows(fetch(base + name + "/profiles/" + name + ".reystress"))
        assert len(means) == len(rey)
        data = []
        for m, r in zip(means, rey):
            # means: y, y+, U, dU/dy, W, dW/dy, P; reystress: y, y+, R_uu,
            # R_vv, R_ww, R_uv, R_uw, R_vw (wall units).
            data.append([m[0], m[1], m[2], math.sqrt(max(r[2], 0.0)),
                         math.sqrt(max(r[3], 0.0)), math.sqrt(max(r[4], 0.0)),
                         r[5]])
        re_tau = {"chan180": "178.12", "chan395": "392.24"}[name]
        write(os.path.join(HERE, "channel", "reference", "mkm_" + name + ".csv"),
              ["Turbulent channel flow, Re_tau = %s: Moser, Kim & Mansour," % re_tau,
               "Phys. Fluids 11 (1999) 943-945 (DNS, Fourier-Chebyshev).",
               "Source: %s%s/profiles/%s.means and .reystress" % (base, name, name),
               "Lower half of the channel, y in units of the half-height h (y = 0",
               "at the wall); everything else in wall units: U+ = U / u_tau,",
               "urms+ = sqrt(R_uu), uv+ = R_uv."],
              ["y", "y_plus", "U_plus", "urms_plus", "vrms_plus", "wrms_plus",
               "uv_plus"], data)


def dfg3d():
    url = "http://www.featflow.de/media/dfg_flow3d/BenchValues.txt"
    data = rows(fetch(url))
    write(os.path.join(HERE, "dfg_3d", "reference", "featflow_3d3z.csv"),
          ["DFG benchmark 3D-3Z (unsteady, Re(t) up to 100): drag and lift",
           "coefficients over t in [0, 8] from FeatFlow's finest computation",
           "(Bayraktar, Mierka & Turek, Int. J. Comput. Sci. Eng. 7 (2012)",
           "253-266). Source: " + url,
           "Their finest-level extremes: c_D,max = 3.2978, c_L,min = -0.010999",
           "(featflow.de, dfg_flow3d results); 3D-1Z: c_D = 6.18533,",
           "c_L = 0.009401."],
          ["t", "c_d", "c_l", "c_z"], data)


def main():
    which = sys.argv[1:] or ["tgv", "channel", "dfg3d"]
    for w in which:
        {"tgv": tgv, "channel": channel, "dfg3d": dfg3d}[w]()


if __name__ == "__main__":
    main()
