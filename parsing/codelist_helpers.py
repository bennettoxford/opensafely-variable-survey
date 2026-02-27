import json
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RSI_JSON_FILE = DATA_DIR / "rsi-codelists-analysis.json"
LATEST_CODELIST_JSON_FILE = ROOT_DIR / "ehrql_codelists_latest_max.json"

_rsi_data = None
_codelists = None


def make_ocl_url(url):
    return f"https://www.opencodelists.org/codelist{url}"


def _get_latest_codelist_data():
    global _codelists
    if _codelists is None:
        _codelists = {}
        with open(LATEST_CODELIST_JSON_FILE) as f:
            json_data = json.load(f)
        for repo_name, commits in json_data.get("projects").items():
            # This is only for the situation where our data file has a single commit
            # (the latest one) from each repo. Let's just fail quickly if that's not right.
            assert len(commits) == 1, (
                f"Expected exactly one commit for repo {repo_name}, but found {len(commits)}"
            )
            _codelists[repo_name] = {
                "codelists": [],
                "inline_codelists": [],
                "unused_codelists": [],
            }
            inline_codelists = dict()
            ocl_codelists = dict()
            for sha, files in commits.items():
                # Remove the "_unused_codelists" key from files before iterating
                for codelist_info in files.pop("_unused_codelists", []):
                    url = codelist_info[0]
                    metadata = lookup_codelist_metadata(url)
                    ocl_url = make_ocl_url(url)
                    _codelists[repo_name]["unused_codelists"].append(
                        {
                            "url": ocl_url,
                            "name": metadata.get("name", "<LOOKUP_FAILED>"),
                            "system": metadata.get("coding_system", "<LOOKUP_FAILED>"),
                        }
                    )

                for file_name, variables in files.items():
                    for variable_name, codelists in variables.items():
                        for codelist_info in codelists:
                            if codelist_info[0] == "<inline>":
                                source_file, line_number = codelist_info[2][7:].split(
                                    ":"
                                )
                                codes = codelist_info[3][7:].split("|")
                                url = f"https://github.com/{repo_name}/blob/{sha}/{source_file}#L{line_number}"
                                inline_codelists[url] = {
                                    "file": source_file,
                                    "line": line_number,
                                    "url": url,
                                    "codes": codes,
                                }
                            elif codelist_info[0]:
                                url = codelist_info[0]
                                metadata = lookup_codelist_metadata(url)
                                ocl_url = make_ocl_url(url)
                                ocl_codelists[url] = {
                                    "url": ocl_url,
                                    "name": metadata.get("name", "<LOOKUP_FAILED>"),
                                    "system": metadata.get(
                                        "coding_system", "<LOOKUP_FAILED>"
                                    ),
                                }

            _codelists[repo_name]["inline_codelists"] = list(inline_codelists.values())
            _codelists[repo_name]["codelists"] = list(ocl_codelists.values())

    return _codelists


def get_repos_with_codelists() -> list[str]:
    codelist_data = _get_latest_codelist_data()
    return list(codelist_data.keys())


def lookup_codelists_by_repo(repo_name: str) -> dict[str, list[dict[str, str]]]:
    codelist_data = _get_latest_codelist_data()
    return codelist_data.get(repo_name, {"codelists": [], "inline_codelists": []})


def _get_rsi_data():
    global _rsi_data
    if _rsi_data is None:
        with open(RSI_JSON_FILE) as f:
            json_data = json.load(f)

        # Build a mapping: version_slug or hash-> (coding_system, full_entry)
        # Each codelist can have multiple versions
        _rsi_data = {}
        for entry in json_data.get("codelists", []):
            coding_system = entry.get("coding_system", "")
            name = entry.get("name", "")
            base_slug = entry.get("slug", "")
            versions = entry.get("versions", [])

            for version in versions:
                tag = version.get("tag")
                hash_val = version.get("hash")

                metadata = {
                    "name": name,
                    "coding_system": coding_system,
                    "creation_method": version.get("creation_method", ""),
                }

                # Create entries for both tag and hash based slugs
                if tag:
                    tag_slug = f"/{base_slug}/{tag}/"
                    _rsi_data[tag_slug] = metadata

                if hash_val:
                    assert hash_val not in _rsi_data, f"Duplicate hash {hash_val}"
                    _rsi_data[hash_val] = metadata

    return _rsi_data


def lookup_codelist_metadata(url_path):
    rsi_data = _get_rsi_data()
    entry = rsi_data.get(url_path)

    if not entry:
        entry = rsi_data.get(url_path.strip("/").split("/")[-1])

    if not entry:
        print(f"Metadata lookup failed for {url_path}")
        return {}

    return entry


def url_to_slug(version_url: str) -> str:
    return version_url.rstrip("/").split("/", 4)[4]


# if __name__ == "__main__":
#     _get_latest_codelist_data()
