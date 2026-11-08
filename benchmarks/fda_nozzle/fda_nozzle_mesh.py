#!/usr/bin/env python3
"""
Structured hexahedral mesh of the FDA benchmark nozzle (Round Robin #1,
sudden-expansion direction), for bessemer (`mesh.geometry: file`).

Geometry (lengths in throat diameters d unless --metres):
  inlet pipe  D = 3d, length L_i (default 4D, as Fehn et al. 2019)
  20-degree (total angle) cone from D to d, length (D - d) / (2 tan 10 deg)
  throat d, length 10d, ending at the sudden expansion, z = 0
  outlet pipe D, length L_o (default 10D): a jet region of --jet-d throat
  diameters with fine, slowly growing cells, then a coarsening far outlet
Flow runs in +z. Sources: Fehn, Wall & Kronbichler, arXiv 1811.06752 Sec. 3;
arXiv 2005.07119 Fig. 1.

Topology: a "butterfly" O-grid. Every cross-section is a square core plus four
ring blocks out to the inner radius (D/2 upstream, shrinking through the cone
to d/2); downstream of the expansion four annulus blocks fill d/2 to D/2, and
their upstream faces are the step wall. Conforming hexes everywhere, no
degenerate cells on the axis, walls exactly circular at --order >= 2.

Physical groups: inlet, outlet, walls (boundary), fluid (volume).

Usage:
  python3 fda_nozzle_mesh.py --level 0 --order 3 -o fda_nozzle_l0.msh
  python3 fda_nozzle_mesh.py --level 1 --order 3 -o fda_nozzle_l1.msh   # ~Fehn l=3
"""
import argparse
import math

import gmsh


def build(args):
    s = 2 ** args.level                       # cells scale per direction
    d = 1.0
    D = 3.0 * d
    L_th = 10.0 * d
    L_c = (D - d) / (2.0 * math.tan(math.radians(10.0)))
    L_i = args.inlet_D * D
    L_o = args.outlet_D * D
    z_in, z_c, z_t, z_e, z_out = -(L_th + L_c + L_i), -(L_th + L_c), -L_th, 0.0, L_o
    z_j = min(args.jet_d * d, 0.8 * L_o)

    nc = args.nc * s                          # cells per quarter circumference
    nr = args.nr * s                          # ring cells, core edge -> wall
    no = args.no * s                          # annulus cells, d/2 -> D/2
    n_ax = dict(inlet=args.n_in * s, cone=args.n_cone * s,
                throat=args.n_throat * s, jet=args.n_jet * s, far=args.n_far * s)
    core = args.core                          # core half-diagonal / radius

    geo = gmsh.model.geo
    angles = [math.pi / 4 + k * math.pi / 2 for k in range(4)]

    def station(z, R, outer=False):
        """Points, curves and surfaces of one cross-section."""
        st = {"z": z}
        c = geo.addPoint(0, 0, z)
        st["sq"] = [geo.addPoint(core * R * math.cos(a), core * R * math.sin(a), z)
                    for a in angles]
        st["ci"] = [geo.addPoint(R * math.cos(a), R * math.sin(a), z) for a in angles]
        st["sq_l"] = [geo.addLine(st["sq"][k], st["sq"][(k + 1) % 4]) for k in range(4)]
        st["ci_l"] = [geo.addCircleArc(st["ci"][k], c, st["ci"][(k + 1) % 4])
                      for k in range(4)]
        st["rad"] = [geo.addLine(st["sq"][k], st["ci"][k]) for k in range(4)]
        if outer:
            st["co"] = [geo.addPoint(D / 2 * math.cos(a), D / 2 * math.sin(a), z)
                        for a in angles]
            st["co_l"] = [geo.addCircleArc(st["co"][k], c, st["co"][(k + 1) % 4])
                          for k in range(4)]
            st["orad"] = [geo.addLine(st["ci"][k], st["co"][k]) for k in range(4)]
        return st

    def faces(st):
        """Cross-section surfaces: core, 4 ring blocks, (4 annulus blocks)."""
        f = {}
        f["core"] = geo.addPlaneSurface([geo.addCurveLoop(st["sq_l"])])
        f["ring"] = []
        for k in range(4):
            k1 = (k + 1) % 4
            loop = geo.addCurveLoop([st["sq_l"][k], st["rad"][k1],
                                     -st["ci_l"][k], -st["rad"][k]])
            f["ring"].append(geo.addSurfaceFilling([loop]))
        if "co" in st:
            f["ann"] = []
            for k in range(4):
                k1 = (k + 1) % 4
                loop = geo.addCurveLoop([st["ci_l"][k], st["orad"][k1],
                                         -st["co_l"][k], -st["orad"][k]])
                f["ann"].append(geo.addSurfaceFilling([loop]))
        return f

    stations = [station(z_in, D / 2), station(z_c, D / 2), station(z_t, d / 2),
                station(z_e, d / 2, outer=True), station(z_j, d / 2, outer=True),
                station(z_out, d / 2, outer=True)]
    sec_faces = [faces(st) for st in stations]

    # Cross-section discretization
    for st in stations:
        for l in st["sq_l"] + st["ci_l"] + st.get("co_l", []):
            geo.mesh.setTransfiniteCurve(l, nc + 1)
        for l in st["rad"]:   # cluster toward the wall
            geo.mesh.setTransfiniteCurve(l, nr + 1, "Progression", args.wall_ratio)
        for l in st.get("orad", []):  # cluster at the shear layer and the wall
            geo.mesh.setTransfiniteCurve(l, no + 1, "Bump", args.bump)

    walls, vols = [], []
    sections = [("inlet", 0, 1, 1.0), ("cone", 1, 2, 1.0),
                ("throat", 2, 3, args.throat_ratio), ("jet", 3, 4, args.jet_ratio),
                ("far", 4, 5, args.far_ratio)]
    for name, a, b, ratio in sections:
        A, B = stations[a], stations[b]
        fa, fb = sec_faces[a], sec_faces[b]
        n = n_ax[name]
        ax = {}
        keys = [("sq", k) for k in range(4)] + [("ci", k) for k in range(4)]
        downstream = name in ("jet", "far")
        if downstream:
            keys += [("co", k) for k in range(4)]
        for key in keys:
            ax[key] = geo.addLine(A[key[0]][key[1]], B[key[0]][key[1]])
            geo.mesh.setTransfiniteCurve(ax[key], n + 1, "Progression", ratio)

        def side(lA, lB, p0, p1):
            loop = geo.addCurveLoop([lA, ax[p1], -lB, -ax[p0]])
            return geo.addSurfaceFilling([loop])

        sq_s = [side(A["sq_l"][k], B["sq_l"][k], ("sq", k), ("sq", (k + 1) % 4))
                for k in range(4)]
        ci_s = [side(A["ci_l"][k], B["ci_l"][k], ("ci", k), ("ci", (k + 1) % 4))
                for k in range(4)]
        rad_s = [side(A["rad"][k], B["rad"][k], ("sq", k), ("ci", k)) for k in range(4)]
        # core
        vols.append(geo.addVolume([geo.addSurfaceLoop(
            [fa["core"], fb["core"]] + sq_s)]))
        for k in range(4):
            k1 = (k + 1) % 4
            vols.append(geo.addVolume([geo.addSurfaceLoop(
                [fa["ring"][k], fb["ring"][k], sq_s[k], rad_s[k1], ci_s[k], rad_s[k]])]))
        if downstream:
            co_s = [side(A["co_l"][k], B["co_l"][k], ("co", k), ("co", (k + 1) % 4))
                    for k in range(4)]
            orad_s = [side(A["orad"][k], B["orad"][k], ("ci", k), ("co", k))
                      for k in range(4)]
            for k in range(4):
                k1 = (k + 1) % 4
                vols.append(geo.addVolume([geo.addSurfaceLoop(
                    [fa["ann"][k], fb["ann"][k], ci_s[k], orad_s[k1], co_s[k], orad_s[k]])]))
            walls += co_s
            if name == "jet":
                walls += fa["ann"]             # the step face at z = 0
        else:
            walls += ci_s

    geo.synchronize()
    for dim, tag in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(2, tag)
    for dim, tag in gmsh.model.getEntities(3):
        gmsh.model.mesh.setTransfiniteVolume(tag)
        gmsh.model.mesh.setRecombine(3, tag)

    inlet = [sec_faces[0]["core"]] + sec_faces[0]["ring"]
    outlet = [sec_faces[5]["core"]] + sec_faces[5]["ring"] + sec_faces[5]["ann"]
    gmsh.model.addPhysicalGroup(2, inlet, 1, "inlet")
    gmsh.model.addPhysicalGroup(2, outlet, 2, "outlet")
    gmsh.model.addPhysicalGroup(2, walls, 3, "walls")
    gmsh.model.addPhysicalGroup(3, vols, 1, "fluid")

    gmsh.model.mesh.generate(3)
    if args.order > 1:
        gmsh.model.mesh.setOrder(args.order)
    if args.metres:
        gmsh.model.mesh.affineTransform([0.004, 0, 0, 0, 0, 0.004, 0, 0, 0, 0, 0.004, 0])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, default=0, help="each level doubles cells per direction")
    p.add_argument("--order", type=int, default=3, help="geometry order (3 for Q3 velocity)")
    p.add_argument("--inlet-D", type=float, default=4.0, help="inlet pipe length / D")
    p.add_argument("--outlet-D", type=float, default=10.0, help="outlet pipe length / D")
    p.add_argument("--nc", type=int, default=8, help="cells per quarter circumference (even)")
    p.add_argument("--nr", type=int, default=6, help="ring cells, core edge to wall")
    p.add_argument("--no", type=int, default=12, help="annulus cells, d/2 to D/2")
    p.add_argument("--n-in", type=int, default=12)
    p.add_argument("--n-cone", type=int, default=8)
    p.add_argument("--n-throat", type=int, default=20)
    p.add_argument("--jet-d", type=float, default=15.0, help="length of the fine jet region / d")
    p.add_argument("--n-jet", type=int, default=60)
    p.add_argument("--n-far", type=int, default=12)
    p.add_argument("--core", type=float, default=0.7, help="core corner radius / local radius")
    p.add_argument("--wall-ratio", type=float, default=0.85, help="ring cell growth toward the wall")
    p.add_argument("--bump", type=float, default=0.25, help="annulus clustering at both ends")
    p.add_argument("--throat-ratio", type=float, default=0.93, help="throat cells shrink toward z=0")
    p.add_argument("--jet-ratio", type=float, default=1.015, help="jet-region cells grow downstream")
    p.add_argument("--far-ratio", type=float, default=1.1, help="far-outlet cells grow downstream")
    p.add_argument("--metres", action="store_true", help="write in metres (d = 4 mm)")
    p.add_argument("-o", "--output", default="fda_nozzle.msh")
    p.add_argument("--msh22", action="store_true", help="write MSH 2.2 instead of 4.1")
    args = p.parse_args()

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("fda_nozzle")
    build(args)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2 if args.msh22 else 4.1)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.write(args.output)
    _, etags, _ = gmsh.model.mesh.getElements(3)
    n_hex = sum(len(t) for t in etags)
    print(f"{args.output}: {n_hex} hexahedra, order {args.order}")
    gmsh.finalize()


if __name__ == "__main__":
    main()
