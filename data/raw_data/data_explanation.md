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
