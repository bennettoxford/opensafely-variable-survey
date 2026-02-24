"""
Build a report of codelists with zero total counts by repository.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from parsing.code_counts import get_counts
from parsing.codelist_cache import get_codelists


DATA_FILE_ROOT = "ehrql_codelists_latest"
DATA_FILE = f"{DATA_FILE_ROOT}.json"
DATA_FILE_MAX = f"{DATA_FILE_ROOT}_max.json"
OUTPUT_FILE_DEFAULT = "data/codelist_zero_counts.json"
RSI_CODELIST_FILE = "data/rsi-codelists-analysis.json"
SUPPORTED_SYSTEMS = {"snomedct", "icd10", "opcs4"}


def load_latest_max() -> dict:
    data_path = Path(DATA_FILE_MAX)
    if not data_path.exists():
        print(
            f"Data file not found: {data_path}\n\nExecute `just codelists --output {DATA_FILE}` to generate it."
        )
        sys.exit(1)
    with data_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_rsi_coding_systems() -> dict[str, str]:
    data_path = Path(RSI_CODELIST_FILE)
    if not data_path.exists():
        print(f"RSI codelist metadata not found: {data_path}")
        sys.exit(1)

    with data_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        data = data["codelists"]

    coding_systems: dict[str, str] = {}
    for entry in data:
        slug = entry.get("slug")
        coding_system = entry.get("coding_system")
        if slug and coding_system:
            coding_systems[slug] = coding_system

        for version in entry.get("versions", []):
            version_slug = version.get("slug")
            if version_slug and coding_system:
                coding_systems[version_slug] = coding_system
            version_hash = version.get("hash")
            if slug and version_hash and coding_system:
                coding_systems[f"{slug}/{version_hash}"] = coding_system

    return coding_systems


def extract_codelists_by_repo() -> dict[str, dict[str, dict | set]]:
    data = load_latest_max()
    codelists_by_repo: dict[str, dict[str, dict | set]] = defaultdict(
        lambda: {"inline": {}, "url": set(), "unknown": set()}
    )

    projects = data.get("projects", {})

    for repo_name, commits in projects.items():
        for commit_sha, files in commits.items():
            for _, variables in files.items():
                for _, codelists in variables.items():
                    for codelist_info in codelists:
                        if not codelist_info or len(codelist_info) == 0:
                            continue

                        first_elem = codelist_info[0]

                        if first_elem is None:
                            continue

                        if first_elem == "<inline>":
                            assert len(codelist_info) >= 4, (
                                f"Unexpected inline format: {codelist_info}"
                            )
                            assert codelist_info[2].startswith("source="), (
                                f"Missing source in inline codelist: {codelist_info}"
                            )
                            assert codelist_info[3].startswith("values="), (
                                f"Missing values in inline codelist: {codelist_info}"
                            )

                            codes = codelist_info[3][len("values=") :]
                            file, line_number = codelist_info[2][
                                len("source=") :
                            ].split(":")
                            permalink = (
                                f"https://github.com/{repo_name}/blob/{commit_sha}/"
                                f"{file}#L{line_number}"
                            )
                            codelists_by_repo[repo_name]["inline"].setdefault(
                                codes, []
                            ).append(permalink)
                        elif (
                            isinstance(first_elem, str)
                            and first_elem.startswith("/")
                            and first_elem.endswith("/")
                        ):
                            codelists_by_repo[repo_name]["url"].add(first_elem)
                        else:
                            codelists_by_repo[repo_name]["unknown"].add(first_elem)

    return codelists_by_repo


def _normalize_codelist_slug(url: str) -> str:
    return url.strip().strip("/")


def _sum_counts(coding_system: str, codes: list[str]) -> int:
    counts = get_counts(coding_system, list(codes))
    return sum(value or 0 for value in counts.values())


def build_zero_event_report() -> dict[str, list[dict[str, object]]]:
    codelists_by_repo = extract_codelists_by_repo()
    rsi_coding_systems = load_rsi_coding_systems()

    zero_by_repo: dict[str, list[dict[str, object]]] = {}

    for repo_name, codelists in codelists_by_repo.items():
        zero_entries: list[dict[str, object]] = []
        for codelist_url, codes in get_codelists(codelists["url"]).items():
            slug = _normalize_codelist_slug(codelist_url)
            coding_system = rsi_coding_systems.get(slug)

            if not coding_system:
                print(f"Can't find coding system for {codelist_url}")
                continue

            if coding_system not in SUPPORTED_SYSTEMS:
                continue

            if not codes:
                continue

            total = _sum_counts(coding_system, codes)
            if total == 0:
                zero_entries.append(
                    {
                        "url": codelist_url,
                        "coding_system": coding_system,
                        "code_count": len(codes),
                        "total_count": total,
                    }
                )
        if zero_entries:
            zero_by_repo[repo_name] = sorted(zero_entries, key=lambda x: x["url"])

    output_path = Path(OUTPUT_FILE_DEFAULT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(zero_by_repo, f, indent=2, sort_keys=True)
        f.write("\n")  # ensure newline at end of file

    return zero_by_repo


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Report codelists with zero total counts by repository."
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the report JSON to stdout",
    )

    args = parser.parse_args()

    report = build_zero_event_report()

    if args.print:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
