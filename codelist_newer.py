import json


with open("notebooks/public/rsi-codelists-analysis.json") as f:
    codelists = json.load(f)["codelists"]


def newer_versions(codelist_version_slug: str) -> list[dict[str, str]]:
    # todo: make use of updated_at whilst taking in to account bulk updates hitting this field
    # i.e. where there is a genuine human-initiated update to the newer version
    codelist_slug, _ = codelist_version_slug.rsplit("/")
    if not (codelist := codelists.get(codelist_slug)):
        raise ValueError(
            f"Could not find codelist {codelist_slug} in opencodelists data dump"
        )
    codelist_version = next(
        v for v in codelist["versions"] if v["slug"] == codelist_version_slug
    )
    published_versions = [v for v in codelist["versions"] if v["status"] == "published"]
    return [
        v
        for v in published_versions
        if v["created_at"] > codelist_version["created_at"]
    ]
