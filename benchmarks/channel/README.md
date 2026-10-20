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

**The decks as committed, at the defaults** (2026-10-19/20; human: the verification
benchmarks "using the defaults we agreed on", no AMR for the channel). Q6/Q5, OIFS at CFL 2
(substeps at CFL 1.5), grad-div γ = 6 h_K / k_u, the decks' solver settings; no overrides.
main a5d8ed0 (the decks) run by the CUDA build of 5da1553 (changes 28-56; the decks change
no code); Bridges-2, one H100 each (GPU-shared), np 1. DNS: Moser, Kim & Mansour (Re_τ
178.12, 392.24); `plot.py`'s errors: U⁺ relative for y⁺ ≥ 5, the stresses the largest
absolute differences in wall units. Peak memory by `nvidia-smi` over the run.

| Deck | Elements | DOFs | t (statistics) | Steps | Wall | Peak memory | Re_τ | U⁺ | u_rms | v_rms | w_rms | u'v' |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `channel_retau180` | 12 × 9 × 12 | 1.02M | 0-120 (60-120) | 41,938 | 17 min | 3.0 GiB | 177.97 | 1.47 % | 0.078 | 0.015 | 0.018 | 0.018 |
| `channel_retau395` | 24 × 15 × 24 | 6.76M | 0-100 (50-100) | 92,014 | 2 h 34 min | 7.9 GiB | 391.72 | **0.66 %** | 0.068 | 0.014 | 0.027 | 0.018 |

The profiles at chosen heights, and the peaks of the stresses (value, y⁺), against the DNS:

| | Re_τ 180 | DNS | Re_τ 395 | DNS |
|---|---|---|---|---|
| U⁺ at y⁺ = 10 | 8.387 (−1.6 %) | 8.522 | 8.342 (−1.2 %) | 8.440 |
| U⁺ at y⁺ = 30 | 13.663 (−1.5 %) | 13.868 | 13.396 (−0.7 %) | 13.493 |
| U⁺ at y⁺ = 100 | 17.103 (−0.3 %) | 17.147 | 16.598 (+0.4 %) | 16.534 |
| U⁺ at the centreline | 18.353 (+0.3 %) | 18.301 | 20.238 (+0.5 %) | 20.133 |
| peak u_rms⁺ | 2.587 (14.7) | 2.658 (15.3) | 2.676 (15.3) | 2.739 (14.2) |
| peak v_rms⁺ | 0.822 (55) | 0.836 (52) | 0.985 (73) | 0.997 (69) |
| peak w_rms⁺ | 1.086 (37) | 1.087 (35) | 1.292 (39) | 1.289 (40) |
| most negative u'v'⁺ | −0.723 (30) | −0.723 (30) | −0.835 (39) | −0.837 (42) |

- **Re_τ within 0.13 % of the DNS in both** (177.97 against 178.12, 391.72 against 392.24):
  the momentum balance holds in the averages, so they are long enough and the walls
  resolved.
- **U⁺ within 1.5 % (Re_τ 180) and 0.7 % (Re_τ 395)** for y⁺ ≥ 5: a little low through the
  buffer layer (y⁺ 10 to 30), a little high at the centreline. The near-wall peak of u_rms⁺
  is 2.7 % and 2.3 % low at the DNS's height; v_rms⁺, w_rms⁺ and u'v'⁺ are within 1.7 % at
  their peaks.
- **Re_τ 395 is the first run of its deck**: 2 h 34 min to t = 100 on one H100 (0.10 s per
  step at 6.76M DOFs), 50 time units of statistics (9,199 samples).
- **Re_τ 180 against the 2026-10-17 Q6 runs below** (deck copies with these settings): U⁺
  1.47 % against 1.05 to 1.68 % then, u_rms 0.078 against 0.043 to 0.061, the other stresses
  alike: the sampling noise of a 60-time-unit average (the runs part in the transition).
  41,938 steps against 42,024; 1,011 s against 1,326 s.

Earlier results, Q3/Q2 decks (until 2026-10-19) and the first Q6 runs:

**Re_τ 180 at Q6/Q5** (2026-10-17, the 3D verification of the GPU memory work): a copy of
the deck with `order_u: 6`, `order_p: 5` and `elements: [12, 9, 12]` (the Q3 deck's
72 × 54 × 72 nodes; the deck's settings since 2026-10-19), OIFS at CFL 2, t = 0 to
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
