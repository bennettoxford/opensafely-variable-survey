"""Extract codelist_from_csv calls from ehrQL variables across GitHub repos.

This script finds ehrQL dataset definition files, parses them with AST to find
all calls to codelist_from_csv for each variable, and outputs the results to
ehrql_codelists.json.

See README.md for usage.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import re
import sys
import time

from parsing.ehrql_github_helpers import (
    clone_repos,
    get_dataset_files,
    get_target_repos_and_shas,
)
from parsing.ehrql_variable_extractor import (
    VariableExtractor,
    extract_variable_codelists,
)


def normalize_path(p: str) -> str:
    """Normalize codelist path for comparison."""
    p = p.replace("\\", "/").strip()
    if p.startswith("./"):
        p = p[2:]
    return p


def parse_codelists_json(repo_root: pathlib.Path) -> tuple[dict[str, str], list[str]]:
    """Parse codelists.json if present and extract URL mapping.

    Supports common layouts:
    - <repo>/codelists/codelists.json (preferred)
    - <repo>/codelists.json (fallback)
    - any other codelists.json found via rglob (last resort)

    Returns:
        Dict mapping normalized codelist path -> URL
    """
    url_map: dict[str, str] = {}
    invalid_slugs: list[str] = []

    # Find candidates, prefer codelists/codelists.json
    all_candidates = list(repo_root.rglob("codelists.json"))
    if not all_candidates:
        return url_map, invalid_slugs

    def candidate_priority(p: pathlib.Path) -> int:
        s = str(p).replace("\\", "/")
        if s.endswith("/codelists/codelists.json"):
            return 0
        if s.endswith("/codelists.json"):
            return 1
        return 2

    all_candidates.sort(key=candidate_priority)
    cf = all_candidates[0]

    try:
        data = json.loads(cf.read_text(encoding="utf-8"))
    except Exception:
        return url_map, invalid_slugs

    # Standard OpenSAFELY structure: { "files": { "<name>.csv": { "url": "...", ... }, ... } }
    # Allowed domain prefixes we'll strip (handle a couple of common variants)
    prefixes = [
        "https://codelists.opensafely.org/codelist",
        "http://codelists.opensafely.org/codelist",
        "https://www.opencodelists.org/codelist",
        "https://opencodelists.org/codelist",
    ]

    def _to_slug(url: str) -> str:
        u = url.strip()
        for p in prefixes:
            if u.startswith(p):
                slug = u[len(p) :]
                # Ensure leading slash
                if not slug.startswith("/"):
                    slug = "/" + slug
                return slug
        # If not matching known prefixes, return the original URL so caller can see it
        return u

    if isinstance(data, dict) and isinstance(data.get("files"), dict):
        for filename, meta in data["files"].items():
            if isinstance(filename, str) and filename.endswith(".csv"):
                if isinstance(meta, dict) and "url" in meta:
                    slug = _to_slug(meta["url"])
                    # Map both "codelists/filename.csv" and just "filename.csv"
                    normalized = normalize_path(f"codelists/{filename}")
                    url_map[normalized] = slug
                    # Also map without codelists/ prefix for flexibility
                    url_map[normalize_path(filename)] = slug
                    # Collect invalid slugs for later reporting/validation
                    # Valid patterns: /user/{username}/{codelist}/{hash_or_tag} OR /{org}/{codelist}/{hash_or_tag}
                    # We'll allow an optional trailing slash
                    if isinstance(slug, str):
                        # If slug still looks like a full URL (didn't match prefixes), treat as invalid
                        if slug.startswith("http://") or slug.startswith("https://"):
                            invalid_slugs.append(slug)
                        else:
                            # Validate allowed slug formats
                            if not re.match(
                                r"^/(user/[^/]+/[^/]+/[^/]+|[^/]+/[^/]+/[^/]+)(?:/)?$",
                                slug,
                            ):
                                invalid_slugs.append(slug)

    return url_map, invalid_slugs


def should_ignore_variable(var_name: str) -> bool:
    """Check if a variable should be ignored in the empty codelists report.

    Args:
        var_name: The variable name to check

    Returns:
        True if the variable should be ignored, False otherwise
    """
    # Ignore list
    ignore_exact = [
        "care_home_tpp",
    ]
    if var_name in ignore_exact:
        return True
    ignore_regex = [
        r"(^|_)sex($|_)",
        r"(^|_)imd($|_)",
        r"(^|_)ethnicity($|_)",
        r"(^|_)region($|_)",
        r"(^|_)death($|_)",
        r"(^|_)died($|_)",
        r"(^|_)dereg($|_)",
        r"(^|_)stp($|_)",
        r"(^|_)registered($|_)",
        r"(^|_)appointment($|_)",
        r"(^|_)alive($|_)",
        r"(^|_)adult($|_)",
        r"(^|_)male($|_)",
        r"(^|_)female($|_)",
        r"(^|_)date.*birth($|_)",
        r"(^|_)rural($|_)",
        r"(^|_)deprivation($|_)",
        r"(^|_)age($|_)",
        r"(^|_)admitted($|_)",
        r"(^|_)registration($|_)",
        r"(^|_)index_date($|_)",
        r"(^|_)dob($|_)",
        r"(^|_)dod($|_)",
        r"(^|_)gp($|_)",
        r"(^|_)practice($|_)",
        r"(^|_)msoa($|_)",
    ]
    for pattern in ignore_regex:
        # Use search so the pattern can match anywhere in the variable name
        # (re.match/re.compile(...).match only tries to match at the start).
        if re.compile(pattern, re.IGNORECASE).search(var_name):
            return True
    return False


def generate_empty_codelists_report(out_map: dict[str, dict]) -> None:
    """Generate and display a report of variables with no codelists found.

    Args:
        out_map: The output map containing project -> files -> variables -> codelists
    """
    empty_vars = []

    # Collect all variables with empty codelists
    for repo, proj_data in out_map.items():
        for file_path, file_vars in proj_data.get("files", {}).items():
            for var_name, codelist_calls in file_vars.items():
                # Check if codelists list is empty and not in ignore list
                if not codelist_calls and not should_ignore_variable(var_name):
                    empty_vars.append((repo, file_path, var_name))

    # Sort for consistent output
    empty_vars.sort()

    # Display report
    print("\n" + "=" * 80, file=sys.stderr)
    print("VARIABLES WITH NO CODELISTS FOUND", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    if not empty_vars:
        print(
            "\n✓ All variables have codelists (or are in the ignore list)",
            file=sys.stderr,
        )
    else:
        # Calculate number of unique variable names (**only counting each once per repo**) and count of each
        var_name_counts_per_repo: dict[str, set[str]] = {}
        for repo, _, var_name in empty_vars:
            repo = repo.split("@")[0] if "@" in repo else repo
            if repo not in var_name_counts_per_repo:
                var_name_counts_per_repo[repo] = set()
            var_name_counts_per_repo[repo].add(var_name)
        var_name_counts: dict[str, int] = {}
        for _, var_names in var_name_counts_per_repo.items():
            for var_name in var_names:
                if var_name not in var_name_counts:
                    var_name_counts[var_name] = 0
                var_name_counts[var_name] += 1

        # Print top 20 most common variable names with empty codelists
        sorted_var_names = sorted(
            var_name_counts.items(), key=lambda x: x[1], reverse=True
        )

        print(
            f"\nTotal: {len(var_name_counts)} variables with empty codelists",
            file=sys.stderr,
        )

        print("\nTop 20 variable names with empty codelists:", file=sys.stderr)
        print("-" * 80, file=sys.stderr)
        for var_name, count in sorted_var_names[:20]:
            print(f"{var_name}: {count} repos", file=sys.stderr)

        # print("\nTop 40:", file=sys.stderr)
        # print("-" * 80, file=sys.stderr)

        # for repo, file_path, var_name in empty_vars:
        #     print(f"{repo} | {file_path} | {var_name}", file=sys.stderr)

        # if len(empty_vars) > 40:
        #     print(f"\n... and {len(empty_vars) - 40} more", file=sys.stderr)

    print("=" * 80 + "\n", file=sys.stderr)


def collect_codelists(
    repos: list[str] | None,
    output: str = "ehrql_codelists.json",
    silent: bool = False,
    verbose: bool = False,
    csv_path: pathlib.Path | None = None,
    force: bool = False,
    use_alt_extractor: bool = False,
) -> None:
    """Collect codelist_from_csv calls for all variables across repositories.

    Args:
        repos: Optional list of repo names to process (e.g., ["opensafely/repo1"])
        output: Output JSON file path
        silent: Suppress all output
        verbose: Verbose progress output to stderr
        csv_path: Optional path to jobs CSV file for repo/SHA combinations
        force: If True, recalculate all results; if False, reuse existing results from output file
        use_alt_extractor: If True, use extract_codelist_calls_alt method instead of default
    """
    initial_start_time = time.time()
    cache_dir = pathlib.Path(".ehrql_repo_cache")
    cache_dir.mkdir(exist_ok=True)

    # Load existing results if not forcing recalculation
    existing_data: dict[str, dict[str, dict]] = {}
    if not force and pathlib.Path(output).exists():
        try:
            with open(output, encoding="utf-8") as f:
                existing_json = json.load(f)
                existing_projects = existing_json.get("projects", {})
                existing_signatures = existing_json.get("signatures", {})

                # Build a map of (repo, sha) -> files_data
                for repo_name, sha_dict in existing_projects.items():
                    if repo_name not in existing_data:
                        existing_data[repo_name] = {}
                    for sha, signature in sha_dict.items():
                        if signature in existing_signatures:
                            existing_data[repo_name][sha] = existing_signatures[
                                signature
                            ]

                if not silent:
                    total_cached = sum(len(shas) for shas in existing_data.values())
                    print(
                        f"Loaded {total_cached} existing SHA results from {output}",
                        file=sys.stderr,
                    )
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            if not silent:
                print(
                    f"Warning: Could not load existing results from {output}: {e}",
                    file=sys.stderr,
                )
            existing_data = {}

    # Get target repos and SHAs (from CSV or GitHub API)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Getting target repos and SHAs")
    target_repos_shas = get_target_repos_and_shas(
        repos=repos if repos else None,
        csv_path=csv_path,
        silent=silent,
        verbose=verbose,
    )

    # Filter out repos/SHAs that are already cached or have no dataset files
    uncached_repos_shas = []
    skipped_no_files = 0
    for repo, sha, list_of_files in target_repos_shas:
        # Skip if we already have results
        if repo in existing_data and sha in existing_data[repo]:
            continue
        # Skip if project_yaml_cache shows this repo@SHA has no dataset files
        if list_of_files is not None and len(list_of_files) == 0:
            skipped_no_files += 1
            continue
        # Need to clone this one
        uncached_repos_shas.append((repo, sha))

    # Report cache filtering
    cached_shas = len(target_repos_shas) - len(uncached_repos_shas) - skipped_no_files
    if not silent:
        print(
            f"Cache filtering: {len(target_repos_shas)} SHAs requested, "
            f"{cached_shas} in output cache, {skipped_no_files} have no files, "
            f"{len(uncached_repos_shas)} to clone",
            file=sys.stderr,
        )

    # Only clone uncached repos
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Cloning uncached ehrQL repos")
    if uncached_repos_shas:
        local_repos = clone_repos(
            uncached_repos_shas,
            repos if repos else None,
            cache_dir,
            silent=silent,
            verbose=verbose,
        )
    else:
        if not silent:
            print("No uncached repos to clone", file=sys.stderr)
        local_repos = []

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Getting dataset files")
    all_dataset_files = get_dataset_files(
        local_repos, silent=silent, verbose=verbose, force=force
    )
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Completed getting dataset files")

    duration = time.time() - initial_start_time
    if not silent:
        print(
            f"\nCompleted repository cloning and dataset file discovery in {duration:.1f}s",
            file=sys.stderr,
        )

    # Structure: project -> sha, files -> {filename -> variables}
    out_map: dict[str, dict] = {}
    cached_count = 0
    processed_count = 0

    # First, add all cached results directly to out_map
    for repo_name, sha_dict in existing_data.items():
        if repo_name not in out_map:
            out_map[repo_name] = {}
        for sha, files_data in sha_dict.items():
            out_map[repo_name][sha] = files_data
            cached_count += 1

    if not silent and cached_count > 0:
        print(
            f"\nAdded {cached_count} cached repo/SHA results to output",
            file=sys.stderr,
        )

    total_shas = len(all_dataset_files)
    # Count unique repos (strip @sha suffix from composite keys)
    unique_repos = set(
        repo.split("@")[0] if "@" in repo else repo for repo in all_dataset_files.keys()
    )
    total_repos = len(unique_repos)
    current_sha_index = 0
    start_time = time.time()

    for repo_key, (head_sha, files, repo_local_dir) in all_dataset_files.items():
        current_sha_index += 1

        # Extract repo name (without @sha suffix)
        repo_name = repo_key.split("@")[0] if "@" in repo_key else repo_key

        if not silent:
            print(
                f"\nProcessing {repo_key} with {len(files)} dataset files... ({current_sha_index}/{total_shas} uncached SHAs, for {total_repos} repos)",
                file=sys.stderr,
            )

        if not files:
            continue

        repo_start_time = time.time()
        variables_processed = 0

        # Build files data structure for this SHA
        files_data = {}

        # Parse codelists.json to get slug mapping for this repo
        resolved_repo_root = repo_local_dir.resolve()
        url_map, invalid_slugs = parse_codelists_json(resolved_repo_root)

        if verbose:
            if url_map:
                print(
                    f"..Found {len(url_map)} codelist entries in codelists.json",
                    file=sys.stderr,
                )
            if invalid_slugs:
                print(
                    f"..Warning: {len(invalid_slugs)} codelist URLs did not match expected slug formats:",
                    file=sys.stderr,
                )
                for s in invalid_slugs:
                    print(f".... {s}", file=sys.stderr)

        for rel_path in files:
            abs_path = repo_local_dir / rel_path
            if not abs_path.exists():
                if verbose:
                    print(
                        f"..File {rel_path} does not exist; skipping", file=sys.stderr
                    )
                continue

            abs_path = abs_path.resolve()

            if verbose:
                print(f"..Extracting codelists from {abs_path}", file=sys.stderr)

            try:
                # Extract codelist calls for all variables in this file
                if use_alt_extractor:
                    extractor = VariableExtractor(abs_path, resolved_repo_root)
                    variable_codelists = extractor.extract_codelist_calls_alt()
                else:
                    variable_codelists = extract_variable_codelists(
                        abs_path, resolved_repo_root
                    )

                if variable_codelists:
                    # Convert to JSON-serializable format and replace paths with URLs
                    file_data = {}
                    for var_name, codelist_calls in variable_codelists.items():
                        # Each codelist_calls is a list of tuples
                        # Convert tuples to lists and replace first param with URL if available
                        processed_calls = []
                        for call in codelist_calls:
                            call_list = list(call)
                            if call_list and call_list[0]:  # Has a first parameter
                                normalized_path = normalize_path(call_list[0])
                                if normalized_path in url_map:
                                    call_list[0] = url_map[normalized_path]
                            processed_calls.append(call_list)

                        file_data[var_name] = processed_calls
                        variables_processed += 1

                    files_data[rel_path] = file_data

                    if verbose:
                        print(
                            f"....Found codelists for {len(variable_codelists)} variables",
                            file=sys.stderr,
                        )

            except SyntaxError as e:
                if not silent:
                    print(
                        f"..Syntax error in {rel_path}: {e}",
                        file=sys.stderr,
                    )
            except Exception as e:
                if not silent:
                    print(
                        f"..Error processing {rel_path}: {e}",
                        file=sys.stderr,
                    )

        # Initialize repo entry if not exists
        if repo_name not in out_map:
            out_map[repo_name] = {}

        # Store files_data for this SHA under this repo
        out_map[repo_name][head_sha] = files_data
        processed_count += 1

        repo_duration = time.time() - repo_start_time
        if not silent:
            print(
                f"..Processed {variables_processed} variables across {len(files)} dataset files in {repo_duration:.1f}s",
                file=sys.stderr,
            )

    duration = time.time() - start_time
    if not silent:
        print(
            f"\nCompleted: {cached_count} cached, {processed_count} processed ({total_shas} total SHAs) in {duration:.1f}s",
            file=sys.stderr,
        )

    # Write output JSON with signature-based deduplication
    write_start_time = time.time()

    # Create signature -> files mapping and repo -> (sha -> signature) mapping
    signatures: dict[str, dict] = {}  # signature -> files data
    sha_to_signature: dict[str, str] = {}  # sha -> signature
    projects: dict[str, dict[str, str]] = {}  # repo -> {sha: signature}

    for repo_name, sha_dict in out_map.items():
        projects[repo_name] = {}
        for sha, files_data in sha_dict.items():
            # Sort files_data for deterministic hashing
            sorted_files = {}
            for file_path in sorted(files_data.keys()):
                file_vars = files_data[file_path]
                # Sort variables by name
                sorted_files[file_path] = {
                    var_name: file_vars[var_name]
                    for var_name in sorted(file_vars.keys())
                }

            # Compute signature (hash of the sorted JSON)
            files_json = json.dumps(sorted_files, sort_keys=True, ensure_ascii=False)
            signature = hashlib.sha256(files_json.encode("utf-8")).hexdigest()[:16]

            # Store mapping
            sha_to_signature[sha] = signature
            projects[repo_name][sha] = signature

            # Store files data by signature (deduplicated)
            if signature not in signatures:
                signatures[signature] = sorted_files

    # Sort projects for deterministic output
    sorted_projects = {repo: projects[repo] for repo in sorted(projects.keys())}

    json_data = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "projects": sorted_projects,
        "signatures": signatures,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")  # Ensure file ends with newline

    # Write a second JSON file without signature deduplication for easier inspection
    # Structure: repo_name > commit_sha > file_name > variable > codelists
    max_output = output.replace(".json", "_max.json")
    max_json_data: dict[str, dict[str, dict[str, dict[str, list]]]] = {}

    for repo_name in sorted(out_map.keys()):
        sha_dict = out_map[repo_name]
        max_json_data[repo_name] = {}
        for sha in sorted(sha_dict.keys()):
            files_data = sha_dict[sha]
            sorted_files = {}
            for file_path in sorted(files_data.keys()):
                file_vars = files_data[file_path]
                sorted_files[file_path] = {
                    var_name: file_vars[var_name]
                    for var_name in sorted(file_vars.keys())
                }
            max_json_data[repo_name][sha] = sorted_files

    with open(max_output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "projects": max_json_data,
            },
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        f.write("\n")  # Ensure file ends with newline

    if not silent:
        write_duration = time.time() - write_start_time
        print(
            f"\nWrote output files in {write_duration:.1f}s",
            file=sys.stderr,
        )

        # Calculate statistics
        total_shas = sum(len(sha_dict) for sha_dict in projects.values())
        total_unique_signatures = len(signatures)
        total_files = sum(len(files_data) for files_data in signatures.values())
        total_variables = sum(
            len(file_vars)
            for files_data in signatures.values()
            for file_vars in files_data.values()
        )
        total_codelist_calls = sum(
            len(calls)
            for files_data in signatures.values()
            for file_vars in files_data.values()
            for calls in file_vars.values()
        )

        print(
            f"\nWrote {output} with {total_codelist_calls} codelist calls "
            f"across {total_variables} variables in {total_files} dataset files "
            f"from {total_shas} SHAs ({total_unique_signatures} unique signatures) "
            f"across {len(projects)} repos",
            file=sys.stderr,
        )
        print(f"Also wrote {max_output} (without deduplication)", file=sys.stderr)

        total_duration = time.time() - initial_start_time
        print(
            f"\nTotal execution time: {total_duration:.1f}s",
            file=sys.stderr,
        )

        # Generate report of variables with no codelists (using signatures)
        # Build a temporary out_map structure for the existing report function
        temp_out_map = {}
        for repo_name, sha_dict in out_map.items():
            for sha, files_data in sha_dict.items():
                key = f"{repo_name}@{sha}"
                temp_out_map[key] = {"sha": sha, "files": files_data}
        generate_empty_codelists_report(temp_out_map)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Extract codelist_from_csv calls from ehrQL variables across GitHub repos"
    )
    p.add_argument(
        "--output",
        default="data/ehrql_codelists.json",
        help="Output JSON file path (default: ehrql_codelists.json)",
    )
    p.add_argument(
        "--silent",
        action="store_true",
        help="Suppress all output",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose progress output to stderr",
    )
    p.add_argument(
        "--csv",
        type=pathlib.Path,
        help="Path to jobs CSV file containing repo URLs and SHAs (uses GitHub API if not provided)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force recalculation of all results, ignoring cached data from previous runs",
    )
    p.add_argument(
        "--use-alt-extractor",
        action="store_true",
        help="Use alternate codelist extraction method (extract_codelist_calls_alt)",
    )
    p.add_argument(
        "repos",
        nargs="*",
        help="Optional list of repo names to process (e.g., opensafely/repo1 opensafely/repo2)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv or sys.argv[1:])

    try:
        collect_codelists(
            repos=args.repos if args.repos else None,
            output=args.output,
            silent=args.silent,
            verbose=args.verbose,
            csv_path=args.csv,
            force=args.force,
            use_alt_extractor=args.use_alt_extractor,
        )
        return 0
    except Exception as e:
        if not args.silent:
            print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
