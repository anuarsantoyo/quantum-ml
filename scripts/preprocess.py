"""Preprocess raw PLE line-width data into a single tidy CSV.

Reads every `.txt` file under data/raw_data/ (two columns: FWHM, fit_error),
parses the experimental metadata from the filename, and concatenates everything
into one long-format CSV at data/processed/fwhm_linewidths.csv.

NaN rows (failed fits) are kept on purpose — NaN handling is left to the EDA.

Usage:
    python scripts/preprocess.py
"""

from pathlib import Path
import re

import pandas as pd

# Repo root = parent of the scripts/ folder this file lives in.
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw_data"
OUT_DIR = ROOT / "data" / "processed"
OUT_FILE = OUT_DIR / "fwhm_linewidths.csv"


def parse_metadata(filename: str) -> dict:
    """Extract experimental parameters from a raw filename.

    Example: fwhm_1nW_240221SIL_Puppy_hindleg_red1nW_Top20nW_Trans05.txt
        -> power_nW=1, transmission=5
    """
    power = re.search(r"red(\d+)nW", filename)
    trans = re.search(r"Trans(\d+)", filename)
    if power is None or trans is None:
        raise ValueError(f"Could not parse power/transmission from {filename!r}")
    return {"power_nW": int(power.group(1)), "transmission": int(trans.group(1))}


def load_file(path: Path) -> pd.DataFrame:
    """Load one raw .txt into a tidy DataFrame, keeping NaN rows."""
    # whitespace-separated, two columns, literal "nan" -> NaN.
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["fwhm", "fit_error"],
        na_values=["nan"],
    )
    meta = parse_metadata(path.name)
    df["power_nW"] = meta["power_nW"]
    df["transmission"] = meta["transmission"]
    df["source_file"] = path.name
    return df


def main() -> None:
    # Skip macOS zip cruft (__MACOSX/, ._* files).
    files = sorted(
        p
        for p in RAW_DIR.rglob("*.txt")
        if "__MACOSX" not in p.parts and not p.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No .txt files found under {RAW_DIR}")

    frames = [load_file(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df = df[["power_nW", "transmission", "fwhm", "fit_error", "source_file"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    valid = df["fwhm"].notna().sum()
    print(f"Read {len(files)} files -> {len(df)} rows ({valid} valid, "
          f"{len(df) - valid} NaN).")
    print(f"Wrote {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
