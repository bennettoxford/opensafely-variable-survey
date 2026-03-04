"""
Helper library to manage code counts from the NHS for 2024-25.

Downloads fixed URLs for the 2024-25 datasets:

Parses to a json files (code_counts.json) stored in the data/ directory.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd
import requests


DOWNLOAD_URLS = {
    "snomedct": "https://files.digital.nhs.uk/9F/527A2C/SNOMED_code_usage_2024-25.txt",
    "icd10": "https://files.digital.nhs.uk/CC/EA025D/hosp-epis-stat-admi-diag-2024-25-tab.xlsx",
    "opcs4": "https://files.digital.nhs.uk/6D/C40538/hosp-epis-stat-admi-proc-2024-25-tab.xlsx",
}

DATA_DIR = Path("data")
COUNT_FILENAME = "code_counts.json"
_COUNTS_CACHE: dict[str, dict[str, int]] = {}


def _fetch_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def _download_if_missing(url: str, dest_dir: Path, force: bool = False) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(url).name
    dest_path = dest_dir / filename
    if dest_path.exists() and not force:
        return dest_path
    dest_path.write_bytes(_fetch_bytes(url))
    return dest_path


def _parse_snomed_counts(path: Path) -> dict[str, int]:
    """Parse SNOMED tab-separated file (2024-25 format)."""
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = list(reader)

    assert rows, f"No data found in SNOMED file: {path}"

    for row in rows[1:]:
        code = row[0].strip()
        val = row[2].strip()
        if val != "*":
            counts[code] = int(val)

    return counts


def _parse_icd10_counts(path: Path) -> dict[str, int]:
    """Parse ICD-10 Excel file (2024-25 format)."""
    df = pd.read_excel(path, sheet_name="All Diagnoses 4 Character")
    counts: dict[str, int] = {}

    for _, row in df.iterrows():
        code = str(row[df.columns[0]]).strip()

        if re.match(r"^[A-Z][0-9]{2}\.[X0-9]$", code):
            val = row[df.columns[7]]
            num_val = 0 if val == "*" else int(val)
            counts[code.replace(".", "")] = num_val
            if "X" in code:
                counts[code.replace(".X", "")] = num_val

    return counts


def _parse_opcs4_counts(path: Path) -> dict[str, int]:
    """Parse OPCS-4 Excel file (2024-25 format)."""
    df = pd.read_excel(path, sheet_name="All Procedure 4 Character")
    counts: dict[str, int] = {}

    for _, row in df.iterrows():
        code = str(row[df.columns[0]]).strip()

        # if matches regex then we add
        if re.match(r"^[A-Z][0-9]{2}\.[X0-9]$", code):
            val = row[df.columns[7]]
            num_val = 0 if val == "*" else int(val)
            counts[code.replace(".", "")] = num_val

    return counts


def download_latest_counts(
    cache_dir: Path, force: bool = False
) -> tuple[Path, Path, Path]:
    """Download the 2024-25 files from fixed URLs."""
    snomed_path = _download_if_missing(DOWNLOAD_URLS["snomedct"], cache_dir, force)
    icd_path = _download_if_missing(DOWNLOAD_URLS["icd10"], cache_dir, force)
    opcs_path = _download_if_missing(DOWNLOAD_URLS["opcs4"], cache_dir, force)
    return (snomed_path, icd_path, opcs_path)


def _build_counts_json(
    paths: tuple[Path, Path, Path],
    output_dir: Path,
) -> None:
    """Parse the three data files and write JSON outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    snomed_path, icd_path, opcs_path = paths
    snomed_counts = _parse_snomed_counts(snomed_path)
    icd_counts = _parse_icd10_counts(icd_path)
    opcs_counts = _parse_opcs4_counts(opcs_path)

    combined_counts = {
        "snomedct": snomed_counts,
        "icd10": icd_counts,
        "opcs4": opcs_counts,
    }

    with (output_dir / COUNT_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(combined_counts, f, indent=2)
        f.write("\n")  # ensure newline at end of file


def _load_counts_for(coding_system: str) -> dict[str, int]:
    normalized = coding_system.lower()

    if normalized in _COUNTS_CACHE:
        return _COUNTS_CACHE[normalized]

    json_path = DATA_DIR / COUNT_FILENAME
    if not json_path.exists():
        cache_dir = DATA_DIR / "code_counts_cache"
        targets = download_latest_counts(cache_dir, force=False)
        _build_counts_json(targets, DATA_DIR)

    if not json_path.exists():
        raise FileNotFoundError(f"Missing counts JSON: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        _COUNTS_CACHE[normalized] = json.load(handle)

    return _COUNTS_CACHE[normalized]


def get_count(coding_system: str, code: str) -> int | None:
    """
    For a given coding system ("snomedct", "icd10", or "opcs4") and code, return the
    count of occurrences in the 2024-25 data.
    Returns 0 if the code is not found or if the count is suppressed.
    """
    counts = _load_counts_for(coding_system)
    return counts[coding_system].get(code, 0)


def get_counts(coding_system: str, codes: list[str]) -> dict[str, int | None]:
    """
    For a given coding system ("snomedct", "icd10", or "opcs4") and list of codes,
    return a dictionary mapping each code to its count of occurrences in the 2024-25 data.
    Returns 0 if the code is not found or if the count is suppressed.
    """
    counts = _load_counts_for(coding_system)
    return {code: counts[coding_system].get(code, 0) for code in codes}
