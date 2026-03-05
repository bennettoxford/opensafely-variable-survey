import json

from parsing.codelist_helpers import url_to_slug


with open("data/rsi-codelists-analysis.json") as f:
    codelists = json.load(f)["codelists"]


def newer_versions(codelist_version_url: str) -> list[dict[str, str]]:
    # `updated_at` might get updated by various automated processes
    # as well as humans making actual updates to the version.
    # the "latest version" behaviour on OpenCodelists uses `created_at`
    # so replicated here.
    if codelist_version_url.endswith(".csv"):
        return []
    codelist_version_slug = url_to_slug(codelist_version_url)
    codelist_slug, _ = codelist_version_slug.rsplit("/", 1)
    if not (
        codelist := next((c for c in codelists if c["slug"] == codelist_slug), None)
    ):
        print(f"Could not find codelist {codelist_slug} in opencodelists data dump")
        return []
    try:
        codelist_version = next(
            v
            for v in codelist["versions"]
            if v["slug"] == codelist_version_slug
            or (codelist_slug + "/" + v["hash"] == codelist_version_slug)
        )
    except StopIteration:
        print(
            f"Could not find version {codelist_version_slug} of codelist - this is because the rsi-codelists-analysis.json data dump likely hasn't been updated since this codelist was added to a repo."
        )
        return []
    return _newer_versions(codelist, codelist_version)


def _newer_versions(codelist: dict, codelist_version: dict) -> list[dict]:
    published_versions = [v for v in codelist["versions"] if v["status"] == "published"]
    return [
        v
        for v in published_versions
        if v["created_at"] > codelist_version["created_at"]
    ]
