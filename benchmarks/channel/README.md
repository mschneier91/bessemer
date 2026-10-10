# Turbulent channel flow, Re_τ 180 and 395

Fully developed channel flow between walls at y = 0 and 2 (δ = 1), periodic in x and z,
driven by a constant mean pressure gradient f_x = 1. That gives u_τ = 1, and the
statistically steady Re_τ is exactly 1/ν. The reference is Moser, Kim & Mansour, Phys.
Fluids 11 (1999) 943-945, at Re_τ 178.12 and 392.24.

| | |
|---|---|
| Reference | `reference/mkm_chan180.csv`, `mkm_chan395.csv`: y, y⁺, U⁺, u_rms⁺, v_rms⁺, w_rms⁺, u'v'⁺ (lower half) |
| Decks | `channel_retau180.yaml` (24 × 18 × 24 elements), `channel_retau395.yaml` (48 × 30 × 48); 2π × 2 × π; OIFS at CFL 2 |
| Start | `initial_velocity: channel`: Reichardt's profile plus a divergence-free perturbation (10% of the centreline velocity) |
| Statistics | `channel_statistics`: plane and time averages at every velocity-node height after `start_time`; wall shear stress, u_τ, Re_τ, bulk velocity; in the run summary (`channel`) and `<path>/<name>_profiles.csv`, rewritten at every checkpoint |
| Compare | `python3 plot.py channel_retau180/chan180_profiles.csv`: folds the halves, then U⁺ and the stresses vs the DNS |

**Checks to make on a run.** The summary's `re_tau` should match `re_tau_target`: the
momentum balance holds exactly in a converged average, so a gap means too short an
average or an under-resolved wall. The transient should be over by `start_time`: compare
the profiles from two checkpoints. Then U⁺ in the log layer, and the peak u_rms⁺ (about
2.7 at y⁺ ≈ 15).

The reference CSV is fetched, not committed: `python3 benchmarks/fetch_references.py`.

Results: none yet.
