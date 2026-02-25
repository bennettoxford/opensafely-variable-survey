import json
from collections.abc import Generator


with open("notebooks/public/rsi-codelists-analysis.json") as f:
    codelists = json.load(f)["codelists"]


def newer_versions(codelist_version_slug: str) -> list[dict[str, str]]:
    # `updated_at` might get updated by various automated processes
    # as well as humans making actual updates to the version.
    # the "latest version" behaviour on OpenCodelists uses `created_at`
    # so replicated here.
    codelist_slug, _ = codelist_version_slug.rsplit("/")
    if not (codelist := next(c for c in codelists if c["slug"] == codelist_slug)):
        raise ValueError(
            f"Could not find codelist {codelist_slug} in opencodelists data dump"
        )
    codelist_version = next(
        v for v in codelist["versions"] if v["slug"] == codelist_version_slug
    )
    return _newer_versions(codelist, codelist_version)


def all_newer_versions() -> Generator[tuple[str, list[dict]]]:
    for codelist in codelists:
        for version in codelist["versions"]:
            yield _newer_versions(codelist, version)


def _newer_versions(codelist: dict, codelist_version: dict) -> list[dict]:
    published_versions = [v for v in codelist["versions"] if v["status"] == "published"]
    return [
        v
        for v in published_versions
        if v["created_at"] > codelist_version["created_at"]
    ]
