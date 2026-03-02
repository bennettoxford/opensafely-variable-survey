import json

from parsing.codelist_helpers import url_to_slug


with open("data/rsi-codelists-analysis.json") as f:
    opencodelists_dump = json.load(f)
    codelists = opencodelists_dump["codelists"]
    releases = opencodelists_dump["releases"]

latest_releases = {
    k: sorted(v, key=lambda v: v["valid_from"], reverse=True)[0]
    for k, v in releases.items()
}


def codelist_version_not_compatible_with_latest_release(
    codelist_version_url: str,
) -> bool | None:
    codelist_version_slug = url_to_slug(codelist_version_url)
    codelist_slug, _ = codelist_version_slug.rsplit("/", 1)
    if not (
        codelist := next((c for c in codelists if c["slug"] == codelist_slug), None)
    ):
        print(f"Could not find codelist {codelist_slug} in opencodelists data dump")
        return False
    codelist_version = next(
        v
        for v in codelist["versions"]
        if v["slug"] == codelist_version_slug
        or (codelist_slug + "/" + v["hash"] == codelist_version_slug)
    )
    coding_system = codelist["coding_system"]
    latest_release = latest_releases[coding_system]["database_alias"]
    latest_release_in_compatible_versions = (
        latest_release in codelist_version["release_compatibility"]
    )
    created_with_latest_release = (
        latest_release == codelist_version["coding_system_release"]
    )
    return not (latest_release_in_compatible_versions or created_with_latest_release)
