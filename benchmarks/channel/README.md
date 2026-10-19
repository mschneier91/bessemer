# Turbulent channel flow, Re_τ 180 and 395

Fully developed channel flow between walls at y = 0 and 2 (δ = 1), periodic in x and z,
driven by a constant mean pressure gradient f_x = 1. That gives u_τ = 1, and the
statistically steady Re_τ is exactly 1/ν. The reference is Moser, Kim & Mansour, Phys.
Fluids 11 (1999) 943-945, at Re_τ 178.12 and 392.24.

| | |
|---|---|
| Reference | `reference/mkm_chan180.csv`, `mkm_chan395.csv`: y, y⁺, U⁺, u_rms⁺, v_rms⁺, w_rms⁺, u'v'⁺ (lower half) |
| Decks | `channel_retau180.yaml` (12 × 9 × 12 Q6/Q5 elements, 72 × 54 × 72 velocity nodes), `channel_retau395.yaml` (24 × 15 × 24, 144 × 90 × 144); 2π × 2 × π; OIFS at CFL 2. Q6/Q5 since 2026-10-19 (the defaults); the Q3/Q2 decks had twice the elements per direction |
| Start | `initial_velocity: channel`: Reichardt's profile plus a divergence-free perturbation (10% of the centreline velocity) |
| Statistics | `channel_statistics`: plane and time averages at every velocity-node height after `start_time`; wall shear stress, u_τ, Re_τ, bulk velocity; in the run summary (`channel`) and `<path>/<name>_profiles.csv`, rewritten at every checkpoint |
| Compare | `python3 plot.py channel_retau180/chan180_profiles.csv`: folds the halves, then U⁺ and the stresses vs the DNS |

**Checks to make on a run.** The summary's `re_tau` should match `re_tau_target`: the
momentum balance holds exactly in a converged average, so a gap means too short an
average or an under-resolved wall. The transient should be over by `start_time`: compare
the profiles from two checkpoints. Then U⁺ in the log layer, and the peak u_rms⁺ (about
2.7 at y⁺ ≈ 15).

The reference CSV is fetched, not committed: `python3 benchmarks/fetch_references.py`.

## Results

**Re_τ 180 at Q6/Q5** (2026-10-17, the 3D verification of the GPU memory work): the deck
with `--set discretization.order_u=6 --set discretization.order_p=5
--set "mesh.elements=[12,9,12]"` (the Q3 deck's 72 × 54 × 72 nodes), OIFS at CFL 2, t = 0 to
120, statistics over t = 60 to 120 (4,190 samples), run with main of before the GPU memory work (aa6aa7e) and main after it (changes 17-27), side by side on Bridges-2, one H100 each. DNS: Moser, Kim & Mansour,
Re_τ 178.12; U⁺ relative for y⁺ >= 5, the stresses in wall units (`plot.py`).

| Run | Re_τ | U⁺ | u_rms | v_rms | w_rms | u'v' | Steps | Wall | Peak memory |
|---|---|---|---|---|---|---|---|---|---|
| main before | 177.87 | 1.17 % | 0.043 | 0.016 | 0.021 | 0.014 | 42,078 | 1,489 s | 3.6 GiB |
| main after | 178.04 | 1.68 % | 0.059 | 0.014 | 0.015 | 0.018 | 42,024 | 1,326 s | 3.0 GiB |
| main after, `time.oifs_on_the_fly` | 177.86 | 1.05 % | 0.061 | 0.014 | 0.014 | 0.014 | 41,925 | 1,399 s | 2.9 GiB |

The runs part in the transition, so their differences (at most 0.14 in U⁺) are the
sampling noise of a 60-time-unit average. At Q3 on the same nodes the production run was
within 7.6 % in U⁺ ([../../docs/throughput.md](../../docs/throughput.md), 2026-10-15).
