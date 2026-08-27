# Raw Data Explanation

Experimental PLE (photoluminescence excitation) line-width data from Dr. Gregor Pieplow.
Format confirmed by Gregor via email (19.06.2026).

## Organization

Two measurement sessions, differing only by the resonant **red-laser power**:

```
data/raw_data/
├── fwhm_1nW_240221/     # red laser @ 1 nW, measured 24 Feb 2021  (7 files)
├── fwhm_3nW_210221/     # red laser @ 3 nW, measured 21 Feb 2021  (7 files)
└── __MACOSX/            # macOS zip cruft — ignore / delete
```

Each session contains 7 `.txt` files sweeping a **transmission setting**:
`Trans05, Trans10, Trans20, Trans40, Trans60, Trans80, Trans100` (= 5 %–100 %).

### Filename tokens

`fwhm_1nW_240221SIL_Puppy_hindleg_red1nW_Top20nW_Trans05.txt`

| Token | Meaning |
|---|---|
| `fwhm` | quantity extracted: line width (FWHM) |
| `1nW` / `240221` | session power label / date (DDMMYY) |
| `SIL` | solid immersion lens (collection optics) |
| `Puppy_hindleg` | sample / scan-spot nickname |
| `red1nW` | resonant red excitation laser power |
| `Top20nW` | top / repump laser power (20 nW) |
| `Trans05 … Trans100` | transmission setting, 5 %–100 % |

## File format

Plain 2-column whitespace table, **3200 rows** per file
(one anomaly: `3nW…Trans40` has 4800 rows). Each row = one fit attempt.

| Column | Meaning |
|---|---|
| 1 | **line width (FWHM)** |
| 2 | **fit error** — 1σ uncertainty on the FWHM (same units as column 1) |
| `nan nan` | fit failed at that row |

Notes:
- **Valid-row count rises with transmission** (~160 at Trans05 → ~2500 at Trans100):
  more signal → more fits succeed. Acts as an effective SNR knob.
- **Median FWHM rises with power** (within and across sessions): power broadening.
- Very large column-2 values coincide with near-zero FWHM — degenerate/failed-ish
  fits where the uncertainty blows up. The dimensionless ratio `err / fwhm` is the
  natural fit-quality metric for filtering.
- The **absolute physical unit** of the line width is not recorded in the files
  (to be confirmed with Gregor; note the MC notebook currently works in MHz).

## Use in the project

The histogram of valid **column-1** values is the experimental FWHM distribution that
the MC pipeline (`simulate → kde → kde_to_bin_counts → L2`) aims to reproduce.
Column 2 (fit error) can later be used to filter low-quality fits and/or weight the loss.


## True Parameters from experiments

All results for both measurement sessions (1 nW and 3 nW red laser power).
Each of the 7 transmission settings maps to a collection efficiency `x_coll_eff`:

| Transmission | `x_coll_eff` |
|---|---|
| `Trans05` | 0.05 |
| `Trans10` | 0.10 |
| `Trans20` | 0.20 |
| `Trans40` | 0.40 |
| `Trans60` | 0.60 |
| `Trans80` | 0.80 |
| `Trans100` | 1.00 |

Fitted parameters per session:

- `pho_normal_mean` — mean of the normal photon-count distribution
- `pho_normal_std` — standard deviation of the normal photon-count distribution
- `pho_noise_poisson_mean` — mean of the Poisson noise distribution

### 1 nW session

| Transmission | `x_coll_eff` | `pho_normal_mean` | `pho_normal_std` | `pho_noise_poisson_mean` |
|---|---|---|---|---|
| `Trans05` | 0.05 | 9.393 | 2.576 | 2.232 |
| `Trans10` | 0.10 | 12.372 | 3.445 | 2.122 |
| `Trans20` | 0.20 | 17.316 | 4.141 | 2.286 |
| `Trans40` | 0.40 | 38.405 | 7.198 | 2.351 |
| `Trans60` | 0.60 | 61.374 | 9.851 | 2.593 |
| `Trans80` | 0.80 | 79.365 | 12.627 | 2.758 |
| `Trans100` | 1.00 | 70.817 | 17.221 | 2.636 |

### 3 nW session

| Transmission | `x_coll_eff` | `pho_normal_mean` | `pho_normal_std` | `pho_noise_poisson_mean` |
|---|---|---|---|---|
| `Trans05` | 0.05 | 13.204 | 3.724 | 2.186 |
| `Trans10` | 0.10 | 24.476 | 5.639 | 2.158 |
| `Trans20` | 0.20 | 34.279 | 8.319 | 2.264 |
| `Trans40` | 0.40 | 84.892 | 24.013 | 2.475 |
| `Trans60` | 0.60 | 103.203 | 23.95 | 2.741 |
| `Trans80` | 0.80 | 137.537 | 32.107 | 2.911 |
| `Trans100` | 1.00 | 175.707 | 40.975 | 3.087 |

### γ from the median (true-γ reference & initialization)

For the 15-series, γ is initialized at — and compared against — the linewidth
estimated from the **median FWHM at Trans100** (x_coll_eff = 1, full collection,
least-broadened measurement of the line):

γ ≈ median(FWHM @ Trans100) / 2   (Lorentzian FWHM = 2γ)

| Session | median FWHM @ Trans100 | γ (median / 2) |
|---|---|---|
| **1 nW** | 17.0 MHz | **8.5 MHz** |
| **3 nW** | 28.3 MHz | **14.1 MHz** |

`σ_phys` (`pho_normal_std`) and λ (`pho_noise_poisson_mean`) remain per-experiment
from the tables above.
