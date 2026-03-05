"""
Script to load the individual json data files for the codelist audit report
and combine them into a single json file.
"""

import json
import os
from datetime import datetime

from codelist_newer_version_available import newer_versions
from codelist_potentially_missing_codes import (
    codelist_version_not_compatible_with_latest_release,
)
from parsing.codelist_helpers import (
    get_repos_with_codelists,
    lookup_codelists_by_repo,
    lookup_latest_releases,
    make_ocl_url,
)


OUTPUT_FILE_DEFAULT = "data/codelist_audit.json"


def load_json_file(file_path):
    with open(file_path) as f:
        return json.load(f)


def main():
    # Load the zero counts json data
    zero_counts_data = load_json_file(os.path.join("data", "codelist_zero_counts.json"))

    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repos": {},
    }

    for repo_name in get_repos_with_codelists():
        repo_output = {
            "good": {"codelists": []},
            "inline_codelists": {"codelists": []},
            "local_codelists": {"codelists": []},
            "unintentional_local_codelists": {"codelists": []},
            "unused_codelists": {"codelists": []},
            "no_events": {"codelists": []},
            "newer_version": {"codelists": []},
            "ethnicity_codelist": False,
            "potentially_missing_codes": {"codelists": []},
        }

        bad_codelists = set()

        # Add no events codelists from the zero counts data
        for codelist in zero_counts_data.get(repo_name, []):
            repo_output["no_events"]["codelists"].append(codelist)
            bad_codelists.add(codelist["url"])

        codelists = lookup_codelists_by_repo(repo_name)

        # Add inline codelists from the inline data
        for inline_codelist in codelists.get("inline_codelists", []):
            repo_output["inline_codelists"]["codelists"].append(inline_codelist)
            bad_codelists.add(inline_codelist.get("url"))

        # add the codelists not using the latest version
        # and codelists with potentially missing codes (i.e. not compatible with latest release)
        for codelist in codelists.get("codelists", []):
            if versions := newer_versions(codelist["url"]):
                latest_version = sorted(
                    versions, key=lambda x: x["created_at"], reverse=True
                )[0]
                newer_version = {
                    "url": make_ocl_url("/" + latest_version["slug"]),
                    "label": latest_version.get("tag") or latest_version["hash"],
                }

                # We want to suggest to users that using the version that already has the human
                # readable labels (instead of the numeric codes) is a good idea. However to guard
                # against potential future releasees of the ethnicity codelist we only flag this in
                # the very specific case where the existing version is 2e641f61 and the newer
                # version is 22911876.
                if codelist["url"].endswith(
                    "opensafely/ethnicity-snomed-0removed/2e641f61/"
                ) and newer_version["url"].endswith("/22911876"):
                    repo_output["ethnicity_codelist"] = {
                        "current_version": codelist["url"],
                        "newer_version": newer_version["url"],
                    }
                else:
                    repo_output["newer_version"]["codelists"].append(
                        codelist | {"newer_version": newer_version}
                    )
                bad_codelists.add(codelist["url"])
            if not codelist["url"].endswith(
                ".csv"
            ) and codelist_version_not_compatible_with_latest_release(codelist["url"]):
                repo_output["potentially_missing_codes"]["codelists"].append(codelist)
                bad_codelists.add(codelist["url"])

        # Add the remaining "good" codelists
        for codelist in codelists.get("codelists", []):
            if codelist.get("url") in bad_codelists:
                continue
            repo_output["good"]["codelists"].append(codelist)

        # Add unused codelists to the output
        repo_output["unused_codelists"]["codelists"] = codelists.get(
            "unused_codelists", []
        )

        # Add local codelists to the output
        repo_output["local_codelists"]["codelists"] = codelists.get(
            "local_codelists", []
        )

        # Add unintentional local codelists to the output
        repo_output["unintentional_local_codelists"]["codelists"] = codelists.get(
            "unintentional_local_codelists", []
        )

        output["repos"][repo_name] = repo_output

    # Add latest releases
    output["latest_releases"] = lookup_latest_releases()

    # Write the output to a json file
    output_file = OUTPUT_FILE_DEFAULT
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"✅ Audit report written to {output_file}")


if __name__ == "__main__":
    main()
