# Flow past a sphere, Re 100-300

Johnson & Patel, J. Fluid Mech. 378 (1999) 19-70.
- **Re ≤ 200:** steady and axisymmetric.
- **210 ≤ Re ≤ 270:** steady, not axisymmetric.
- **Re 300:** periodic hairpin shedding, with C_D = 0.656, C_L = 0.069, St = 0.137.

**Not set up yet: it needs a mesh.** bessemer reads hexahedral meshes from Gmsh or MFEM
files (`mesh.geometry: file`) and takes the boundary names from the Gmsh physical groups.
Gmsh itself isn't in the Spack stack. Make a hex mesh of a sphere (diameter 1) in a box,
with physical groups `inflow`, `outflow`, `sides` and `sphere`. Then a deck along these
lines runs it:

```yaml
equation: navier_stokes
physics: {nu: 3.333333e-3}            # Re = U D / nu = 300
mesh: {dim: 3, geometry: file, file: sphere.msh}
time: {dt: 0.01, t_final: 300, step_control: cfl, cfl_target: 2.0, convection: oifs}
boundary_conditions:
  - {select: [inflow, sides], type: velocity, value: [1, 0, 0]}
  - {select: [outflow], type: outflow}
  - {select: [sphere], type: no_slip}
forces: {enabled: true, boundaries: [sphere], reference_velocity: 1.0,
         reference_area: 0.785398, statistics: true}      # pi D^2 / 4
amr: {enabled: true, every_time: 1.0}  # refine the wake as it develops
reference: {cd_mean: 0.656}
```

Two features still to add: the summary's shedding statistics report a Strouhal number in
2D only, and the lift in 3D sheds in a plane that isn't fixed in advance.
