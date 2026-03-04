"""
Build a report of codelists with zero total counts by repository.
"""

import json
from pathlib import Path

from parsing.code_counts import get_counts
from parsing.codelist_cache import get_codelist
from parsing.codelist_helpers import (
    get_repos_with_codelists,
    lookup_codelists_by_repo,
)


OUTPUT_FILE_DEFAULT = "data/codelist_zero_counts.json"
SUPPORTED_SYSTEMS = {"snomedct", "icd10", "opcs4"}


def _sum_counts(coding_system: str, codes: list[str]) -> int:
    counts = get_counts(coding_system, list(codes))
    return sum(value or 0 for value in counts.values())


def build_zero_event_report() -> dict[str, list[dict[str, object]]]:

    zero_by_repo: dict[str, list[dict[str, object]]] = {}

    for repo_name in get_repos_with_codelists():
        codelists = lookup_codelists_by_repo(repo_name)["codelists"]
        zero_entries: list[dict[str, object]] = []
        for codelist in codelists:
            coding_system = codelist["system"]
            codelist_url = codelist["url"]
            codes = get_codelist(codelist_url)

            if not coding_system:
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
                        "system": coding_system,
                        "code_count": len(codes),
                        "total_count": total,
                        "name": codelist["name"],
                        "variables": codelist["variables"],
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
