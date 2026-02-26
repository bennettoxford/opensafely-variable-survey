"""
Helpers for fetching and caching codelists from opencodelists.org.
"""

from __future__ import annotations

import csv
from pathlib import Path
from urllib.parse import quote

import requests


CACHE_DIR = Path(".codelist_cache")
BASE_URL = "https://www.opencodelists.org/codelist"


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(exist_ok=True)


def _get_cache_path(codelist_url: str) -> Path:
    safe_name = quote(codelist_url.strip("/"), safe="")
    return CACHE_DIR / f"{safe_name}.csv"


def _get_codes_from_file(cache_path: Path) -> list[str]:
    with cache_path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)  # skip header
        return [row[0] for row in reader]


def normalize_codelist_url(url: str) -> str:
    return url.replace("https://www.opencodelists.org/codelist/", "").strip("/")


def get_codelist(codelist_url: str, force: bool = False) -> list[str]:
    """
    Get codelist from cache or download it if not cached or if force is True. Returns list of clinical codes.
    """
    _ensure_cache_dir()
    url = normalize_codelist_url(codelist_url)
    cache_path = _get_cache_path(url)

    if not force and cache_path.exists():
        return _get_codes_from_file(cache_path)

    download_url = f"{BASE_URL}{url}download.csv"
    try:
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return None

    cache_path.write_bytes(response.content)

    return _get_codes_from_file(cache_path)


def get_codelists(
    codelist_urls: list[str], force: bool = False
) -> dict[str, list[str]]:
    codelists: dict[str, list[str]] = {}
    for url in codelist_urls:
        codelist = get_codelist(url, force=force)
        if codelist is not None:
            codelists[url] = codelist
        else:
            print(f"Warning: Failed to fetch codelist for {url}")
    return codelists
