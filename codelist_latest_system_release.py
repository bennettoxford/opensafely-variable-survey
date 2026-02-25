import json
from collections.abc import Generator


with open("notebooks/public/rsi-codelists-analysis.json") as f:
    opencodelists_dump = json.load(f)
    codelists = opencodelists_dump["codelists"]
    releases = opencodelists_dump["releases"]

latest_releases = {
    k: sorted(v, key=lambda v: v["valid_from"], reverse=True)[0] for k, v in releases
}


def codelist_version_compatible_with_latest_release(version_slug: str) -> bool:
    codelist_slug = version_slug.rsplit("/")[0]
    if not (codelist := next(c for c in codelists if c["slug"] == codelist_slug)):
        raise ValueError(
            f"Could not find codelist {codelist_slug} in opencodelists data dump"
        )
    version = next(v for v in codelist["versions"] if v["slug"] == version_slug)
    coding_system = codelist["coding_system"]
    return _version_compatible_with_latest_release(coding_system, version)


def all_codelist_version_compatible_with_latest_release() -> Generator[
    tuple[str, bool]
]:
    for codelist in codelists:
        coding_system = codelist["coding_system"]
        for version in codelist["versions"]:
            yield (
                version["slug"],
                _version_compatible_with_latest_release(coding_system, version),
            )


def _version_compatible_with_latest_release(coding_system: str, version: dict) -> bool:
    return (
        latest_releases[coding_system]["database_alias"]
        in version["release_compatibility"]
    )
