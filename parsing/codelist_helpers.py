import json
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RSI_JSON_FILE = DATA_DIR / "rsi-codelists-analysis.json"
LATEST_CODELIST_JSON_FILE = ROOT_DIR / "ehrql_codelists_latest_max.json"
METADATA_CACHE_FILE = DATA_DIR / "codelist_metadata_cache.json"

_rsi_data = None
_codelists = None
_metadata_cache = None
_empty_cache_retry_attempted = set()


def make_ocl_url(url):
    return f"https://www.opencodelists.org/codelist{url}"


def make_github_url(repo_name, filepath, sha, line_number=None):
    url = f"https://github.com/{repo_name}/blob/{sha}/{filepath}"
    if line_number:
        url += f"#L{line_number}"
    return url


def _get_metadata_cache():
    global _metadata_cache
    if _metadata_cache is None:
        if METADATA_CACHE_FILE.exists():
            with open(METADATA_CACHE_FILE) as f:
                _metadata_cache = json.load(f)
        else:
            _metadata_cache = {}

    return _metadata_cache


def _persist_metadata_cache():
    cache = _get_metadata_cache()
    with open(METADATA_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
        f.write("\n")


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
                "local_codelists": [],
                "unintentional_local_codelists": [],
            }
            inline_codelists = dict()
            ocl_codelists = dict()
            local_codelists = dict()
            unintentional_local_codelists = dict()
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
                            "system_release": metadata.get(
                                "coding_system_release", "<LOOKUP_FAILED>"
                            ),
                        }
                    )

                for file_name, variables in files.items():
                    for variable_name, codelists in variables.items():
                        for codelist_info in codelists:
                            variable_content = {
                                "variables": [variable_name],
                                "github_url": make_github_url(
                                    repo_name, file_name, sha
                                ),
                            }
                            variable_object = {}
                            variable_object[file_name] = variable_content
                            if codelist_info[0] == "<inline>":
                                source_file, line_number = codelist_info[2][7:].split(
                                    ":"
                                )
                                codes = codelist_info[3][7:].split("|")
                                url = f"https://github.com/{repo_name}/blob/{sha}/{source_file}#L{line_number}"
                                if url in inline_codelists:
                                    if file_name in inline_codelists[url]["variables"]:
                                        inline_codelists[url]["variables"][file_name][
                                            "variables"
                                        ].append(variable_name)
                                    else:
                                        inline_codelists[url]["variables"][
                                            file_name
                                        ] = variable_content
                                else:
                                    inline_codelists[url] = {
                                        "file": source_file,
                                        "line": line_number,
                                        "url": url,
                                        "codes": codes,
                                        "variables": variable_object,
                                    }
                            elif codelist_info[0] and codelist_info[0].endswith(".csv"):
                                url = codelist_info[0]
                                if url.startswith("codelists/"):
                                    if url in unintentional_local_codelists:
                                        if (
                                            file_name
                                            in unintentional_local_codelists[url][
                                                "variables"
                                            ]
                                        ):
                                            unintentional_local_codelists[url][
                                                "variables"
                                            ][file_name]["variables"].append(
                                                variable_name
                                            )
                                        else:
                                            unintentional_local_codelists[url][
                                                "variables"
                                            ][file_name] = variable_content
                                    else:
                                        unintentional_local_codelists[url] = {
                                            "file": url,
                                            "url": make_github_url(repo_name, url, sha),
                                            "variables": variable_object,
                                        }
                                else:
                                    if url in local_codelists:
                                        if (
                                            file_name
                                            in local_codelists[url]["variables"]
                                        ):
                                            local_codelists[url]["variables"][
                                                file_name
                                            ]["variables"].append(variable_name)
                                        else:
                                            local_codelists[url]["variables"][
                                                file_name
                                            ] = variable_content
                                    else:
                                        local_codelists[url] = {
                                            "file": url,
                                            "url": make_github_url(repo_name, url, sha),
                                            "variables": variable_object,
                                        }
                            elif codelist_info[0]:
                                url = codelist_info[0]
                                metadata = lookup_codelist_metadata(url)
                                ocl_url = make_ocl_url(url)
                                if url in ocl_codelists:
                                    if file_name in ocl_codelists[url]["variables"]:
                                        ocl_codelists[url]["variables"][file_name][
                                            "variables"
                                        ].append(variable_name)
                                    else:
                                        ocl_codelists[url]["variables"][file_name] = (
                                            variable_content
                                        )
                                else:
                                    ocl_codelists[url] = {
                                        "url": ocl_url,
                                        "name": metadata.get("name", "<LOOKUP_FAILED>"),
                                        "system": metadata.get(
                                            "coding_system", "<LOOKUP_FAILED>"
                                        ),
                                        "system_release": metadata.get(
                                            "coding_system_release", "<LOOKUP_FAILED>"
                                        ),
                                        "variables": variable_object,
                                    }

            _codelists[repo_name]["inline_codelists"] = list(inline_codelists.values())
            _codelists[repo_name]["local_codelists"] = list(local_codelists.values())
            _codelists[repo_name]["unintentional_local_codelists"] = list(
                unintentional_local_codelists.values()
            )
            _codelists[repo_name]["codelists"] = list(ocl_codelists.values())

    return _codelists


def get_repos_with_codelists() -> list[str]:
    codelist_data = _get_latest_codelist_data()
    return list(codelist_data.keys())


def lookup_codelists_by_repo(repo_name: str) -> dict[str, list[dict[str, str]]]:
    codelist_data = _get_latest_codelist_data()
    return codelist_data.get(
        repo_name,
        {
            "codelists": [],
            "inline_codelists": [],
            "unused_codelists": [],
            "local_codelists": [],
            "unintentional_local_codelists": [],
        },
    )


def _get_rsi_data():
    global _rsi_data
    if _rsi_data is None:
        with open(RSI_JSON_FILE) as f:
            json_data = json.load(f)

        # Build a mapping: version_slug or hash-> (coding_system, full_entry)
        # Each codelist can have multiple versions
        _rsi_data = {"codelist_versions": {}, "latest_releases": {}}

        for coding_system, releases in json_data["releases"].items():
            # Get latest release
            latest_release = sorted(
                releases, key=lambda x: x["valid_from"], reverse=True
            )[0]
            _rsi_data["latest_releases"][coding_system] = latest_release["release_name"]

        for entry in json_data.get("codelists", []):
            coding_system = entry.get("coding_system", "")
            name = entry.get("name", "")
            base_slug = entry.get("slug", "")
            versions = entry.get("versions", [])

            for version in versions:
                tag = version.get("tag")
                hash_val = version.get("hash")
                coding_system_release = version.get("coding_system_release", "")

                metadata = {
                    "name": name,
                    "coding_system": coding_system,
                    "coding_system_release": coding_system_release,
                    "creation_method": version.get("creation_method", ""),
                }

                # Create entries for both tag and hash based slugs
                if tag:
                    tag_slug = f"/{base_slug}/{tag}/"
                    _rsi_data["codelist_versions"][tag_slug] = metadata

                if hash_val:
                    assert hash_val not in _rsi_data["codelist_versions"], (
                        f"Duplicate hash {hash_val}"
                    )
                    _rsi_data["codelist_versions"][hash_val] = metadata

    return _rsi_data


def lookup_codelist_metadata(url_path):
    if url_path.endswith(".csv"):
        # We only want to lookup URLs - throw an error so we catch this quickly
        raise ValueError(
            f"URL path {url_path} looks like a direct CSV link, skipping metadata lookup"
        )

    rsi_data = _get_rsi_data()["codelist_versions"]
    entry = rsi_data.get(url_path)

    if not entry:
        entry = rsi_data.get(url_path.strip("/").split("/")[-1])

    if not entry:
        metadata_cache = _get_metadata_cache()
        if url_path in metadata_cache:
            cached_entry = metadata_cache[url_path]
            if cached_entry:
                return cached_entry
            if url_path in _empty_cache_retry_attempted:
                return cached_entry
            _empty_cache_retry_attempted.add(url_path)
        # It might be an issue with a codelist with multiple handles
        # Let's query OCL and see if we get a 302
        ocl_url = make_ocl_url(url_path)

        # the redirect only happens if you don't specify a tag or hash
        ocl_url_no_version = ocl_url.rstrip("/").rsplit("/", 1)[0] + "/"
        try:
            import requests

            response = requests.head(
                ocl_url_no_version, allow_redirects=True, timeout=5
            )
            if response.status_code == 200 and response.url != ocl_url_no_version:
                redirected_slug = response.url.split("/codelist/")[1]
                entry = rsi_data.get("/" + redirected_slug, {})
                metadata_cache[url_path] = entry
                _persist_metadata_cache()
        except Exception as e:
            print(f"Error occurred while fetching OCL data for {url_path}: {e}")
            metadata_cache[url_path] = {}
            _persist_metadata_cache()
            return {}

        if not entry:
            print(f"Metadata lookup failed for {url_path}")
            metadata_cache[url_path] = {}
            _persist_metadata_cache()
            return {}

    return entry


def lookup_latest_releases():
    rsi_data = _get_rsi_data()
    return rsi_data["latest_releases"]


def url_to_slug(version_url: str) -> str:
    return version_url.rstrip("/").split("/", 4)[4]


# if __name__ == "__main__":
#     _get_latest_codelist_data()
