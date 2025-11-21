import base64
import csv
import json
import pathlib
import re
import subprocess
import sys
import time

import yaml


GH_API_HEADERS = [
    "Accept: application/vnd.github+json",
    "X-GitHub-Api-Version: 2022-11-28",
]


class GitHubError(RuntimeError):
    """Generic error for GitHub CLI interactions."""

    pass


def run_gh(args: list[str], expect_json: bool = True) -> dict | list | str:
    """
    Run a gh CLI command with given args and return parsed JSON output or raw string.
    """
    cmd = ["gh"] + args
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise GitHubError(
            "gh CLI not found. Install from https://cli.github.com/"
        ) from e
    if proc.returncode != 0:
        raise GitHubError(f"gh command failed: {' '.join(cmd)}\n{proc.stderr.strip()}")
    if not expect_json:
        return proc.stdout
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise GitHubError(
            f"Failed to parse JSON from gh output for command: {' '.join(cmd)}\nOutput: {proc.stdout[:500]}"
        ) from e


def fetch_file_content(owner: str, repo: str, path: str, ref: str) -> tuple[str, str]:
    """Fetch file content from GitHub repo at given ref."""
    data = run_gh(
        [
            "api",
            f"repos/{owner}/{repo}/contents/{path}?ref={ref}",
        ]
    )
    if isinstance(data, list):  # directory edge case, skip
        return "", ""
    encoding = data.get("encoding")
    if encoding == "base64":
        try:
            content = base64.b64decode(data.get("content", "")).decode(
                "utf-8", "replace"
            )
        except Exception:
            content = ""
    else:
        content = data.get("content", "")
    return content, data.get("sha", "")


def parse_project_yaml(repo_root: pathlib.Path) -> list[str]:
    """Extract dataset definition file paths from project.yaml.

    Returns list of relative paths to files that generate datasets.
    """
    try:
        content = (repo_root / "project.yaml").read_text(encoding="utf-8")
        data = yaml.safe_load(content)

        dataset_files = set()
        if not data:
            return list(dataset_files)
        actions = data.get("actions", {})

        for action_name, action_config in actions.items():
            # Look for generate_dataset commands
            run_command = action_config.get("run", "")
            if "generate-dataset" in run_command or "generate_dataset" in run_command:
                filtered_command = re.sub(r"--test-data-file\s+\S+", "", run_command)
                # Extract file path from command like "ehrql:v1 generate-dataset analysis/dataset_definition.py"
                # Split by whitespace and look for .py files. File names may be wrapped in quotes, so we strip
                # them off.
                parts = [p.strip("\"'") for p in filtered_command.split()]

                possible_files = [p for p in parts if p.endswith(".py")]
                if len(possible_files) == 1:
                    dataset_files.add(possible_files[0])
                else:
                    print(
                        f"..Warning: Could not unambiguously extract dataset file from command: {run_command}\n"
                        f"... in {repo_root / 'project.yaml'}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

        return list(dataset_files)
    except (GitHubError, yaml.YAMLError, KeyError) as _:
        return []


def parse_jobs_csv(csv_path: pathlib.Path) -> dict[str, list[str]]:
    """Parse jobs CSV and extract repo -> list of SHAs mapping.

    Args:
        csv_path: Path to the jobs CSV file

    Returns:
        Dict mapping repo full name (e.g., 'opensafely/repo-name') to list of unique SHAs
    """
    repo_shas: dict[str, set[str]] = {}

    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("url", "").strip()
                sha = row.get("sha", "").strip()

                if not url or not sha:
                    continue

                # Extract repo name from URL like "https://github.com/opensafely/repo-name"
                if url.startswith("https://github.com/"):
                    repo_full_name = url.replace("https://github.com/", "").strip("/")
                    if "/" in repo_full_name:
                        if repo_full_name not in repo_shas:
                            repo_shas[repo_full_name] = set()
                        repo_shas[repo_full_name].add(sha)
    except FileNotFoundError:
        print(f"Warning: CSV file not found: {csv_path}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Warning: Error parsing CSV: {e}", file=sys.stderr)
        return {}

    # Convert sets to sorted lists for deterministic output
    return {repo: sorted(shas) for repo, shas in sorted(repo_shas.items())}


def project_yaml_search(verbose: bool = False) -> list[dict]:
    """Search GitHub for project.yaml files in opensafely ehrql repos."""
    query = "org:opensafely+ehrql+filename:project.yaml"
    items: list[dict] = []
    page = 1
    while True:
        header_args: list[str] = []
        for h in GH_API_HEADERS:
            header_args.extend(["-H", h])
        path_with_query = f"/search/code?q={query}&per_page=100&page={page}"
        result = run_gh(["api", *header_args, path_with_query])
        batch = result.get("items", []) if isinstance(result, dict) else []
        for it in batch:
            repo = it.get("repository", {})
            full_name = repo.get("full_name")
            if repo["private"]:
                if verbose:
                    print(f"Skipping private repo {full_name}")
                continue
            if full_name:
                it["repo_full_name"] = full_name
                items.append(it)
        if len(batch) < 100:
            break
        page += 1
        if page > 10:  # safety guard (100 * 10 = 1000 limit)
            break
    return items


def group_items_by_repo(items: list[dict]) -> dict:
    grouped: dict = {}
    for it in items:
        repo = it.get("repo_full_name")
        head_sha = it.get("html_url").split("/")[6]
        if not repo:
            continue
        # Add key of repo and value of sha. If it already exists, log if different
        existing = grouped.get(repo, None)
        if existing and existing != head_sha:
            print(
                f"Warning: Different head SHA found for {repo}: {existing} vs {head_sha}"
            )
        grouped[repo] = head_sha
    return grouped


def get_remote_head_shas(repos: list[str], verbose: bool = False) -> dict[str, str]:
    """Get the current HEAD SHA for multiple repos via GitHub GraphQL API.

    This is much faster than doing git fetch for each repo individually.
    Returns dict mapping repo_full_name -> HEAD SHA.
    """
    if not repos:
        return {}

    # Build GraphQL query for all repos at once
    # Format: repo1: repository(owner: "opensafely", name: "repo1") { defaultBranchRef { target { oid } } }
    repo_queries = []
    repo_aliases = {}

    for i, repo_full_name in enumerate(repos):
        owner, repo_name = repo_full_name.split("/", 1)
        alias = f"repo{i}"
        repo_aliases[alias] = repo_full_name
        repo_queries.append(
            f'{alias}: repository(owner: "{owner}", name: "{repo_name}") {{ defaultBranchRef {{ target {{ oid }} }} }}'
        )

    query = "query { " + " ".join(repo_queries) + " }"

    try:
        result = run_gh(["api", "graphql", "-f", f"query={query}"])

        head_shas = {}
        if isinstance(result, dict) and "data" in result:
            for alias, repo_full_name in repo_aliases.items():
                repo_data = result["data"].get(alias)
                if repo_data and repo_data.get("defaultBranchRef"):
                    sha = repo_data["defaultBranchRef"]["target"]["oid"]
                    head_shas[repo_full_name] = sha

        return head_shas
    except GitHubError as e:
        if verbose:
            print(
                f"Warning: Failed to fetch remote HEADs via GraphQL: {e}",
                file=sys.stderr,
            )
        return {}


def clone_repos(
    all_repos: tuple[str, str],
    repos: list[str],
    cache_dir: pathlib.Path,
    silent: bool = False,
    verbose: bool = False,
) -> tuple[str, str, str]:
    """Clone or update GitHub repos to local base_dir using worktrees.

    For each repo:
    - Creates a bare clone in {repo_name}/ (or updates if exists)
    - Creates worktrees for each SHA in {repo_name}-{sha}/
    - Fetches latest from origin only if local HEAD differs from remote

    repos: tuple of (full_repo_name, ref_sha)
    Returns list of tuples (repo_full_name, ref_sha, local_path).
    """
    # Group SHAs by repo
    repo_sha_map: dict[str, list[str]] = {}
    for repo_full_name, ref_sha in all_repos:
        if (
            repos
            and repo_full_name not in repos
            and repo_full_name.split("/", 1)[1] not in repos
        ):
            continue
        if repo_full_name not in repo_sha_map:
            repo_sha_map[repo_full_name] = []
        repo_sha_map[repo_full_name].append(ref_sha)

    # Get remote HEAD SHAs for all repos in one batch API call
    print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Getting remote HEAD SHAs for ehrQL repos"
    )
    remote_heads = get_remote_head_shas(list(repo_sha_map.keys()), verbose=verbose)
    print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Completed getting remote HEAD SHAs for ehrQL repos"
    )
    local_repos = []

    for repo_full_name, shas in repo_sha_map.items():
        owner, repo_name = repo_full_name.split("/", 1)

        if not silent:
            unique_shas = list(set(shas))
            print(
                f"\n==> {repo_full_name} ({len(unique_shas)} unique SHA(s))",
                file=sys.stderr,
            )

        remote_head_sha = remote_heads.get(repo_full_name)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Cloning {repo_full_name}")
        cloned_repos = _clone_with_worktrees(
            repo_full_name,
            owner,
            repo_name,
            shas,
            cache_dir,
            remote_head_sha,
            silent,
            verbose,
        )
        local_repos.extend(cloned_repos)

    return local_repos


def _clone_with_worktrees(
    repo_full_name: str,
    owner: str,
    repo_name: str,
    shas: list[str],
    cache_dir: pathlib.Path,
    remote_head_sha: str | None,
    silent: bool,
    verbose: bool,
) -> list[tuple[str, str, pathlib.Path]]:
    """Clone a repo as bare and create worktrees for each SHA.

    On first run: creates bare clone
    On subsequent runs: only fetches if remote HEAD differs from local HEAD

    Args:
        remote_head_sha: The current HEAD SHA from GitHub API (if available)

    Returns list of tuples (repo_full_name, sha, worktree_path).
    """
    # Main bare repo directory (no suffix, just the repo name)
    bare_repo_dir = cache_dir / repo_name
    clone_url = f"https://github.com/{owner}/{repo_name}.git"

    # Do bare clone if not already cached, otherwise check if fetch needed
    if not bare_repo_dir.exists():
        cmd = ["git", "clone", "--bare", clone_url, str(bare_repo_dir)]

        if verbose:
            print(
                f"..Creating bare clone of {repo_full_name} at {bare_repo_dir}",
                file=sys.stderr,
            )

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            if not silent:
                print(
                    f"..Bare clone failed {owner}/{repo_name}: {proc.stderr.strip()}",
                    file=sys.stderr,
                )
            return []
    else:
        # Repo exists, check if we need to fetch
        # If we have the remote HEAD SHA, check if its worktree exists
        needs_fetch = False

        if remote_head_sha:
            # Check if worktree for remote HEAD exists - if so, we're up to date
            remote_head_worktree = cache_dir / f"{repo_name}-{remote_head_sha[:8]}"
            if not remote_head_worktree.exists():
                # Remote HEAD worktree doesn't exist, need to fetch
                needs_fetch = True
                if verbose:
                    print(
                        "..Remote HEAD worktree not found, fetching updates",
                        file=sys.stderr,
                    )
        else:
            # No remote HEAD info from API, fall back to fetch (rare case)
            needs_fetch = True
            if verbose:
                print(
                    "..No remote HEAD info available, fetching to be safe",
                    file=sys.stderr,
                )

        if needs_fetch:
            fetch_proc = subprocess.run(
                ["git", "--git-dir", str(bare_repo_dir), "fetch", "origin"],
                capture_output=True,
                text=True,
            )
            if fetch_proc.returncode != 0 and not silent:
                print(
                    f"..Warning: fetch failed for {repo_full_name}: {fetch_proc.stderr.strip()}",
                    file=sys.stderr,
                )
        elif verbose:
            print(
                "..Remote HEAD worktree exists, skipping fetch",
                file=sys.stderr,
            )

    # Create worktrees for each SHA
    local_repos = []
    for sha in shas:
        worktree_dir = cache_dir / f"{repo_name}-{sha[:8]}"

        if not worktree_dir.exists():
            # First, try to ensure we have this commit
            # Check if commit exists, if not try to fetch it
            check_proc = subprocess.run(
                ["git", "--git-dir", str(bare_repo_dir), "cat-file", "-e", sha],
                capture_output=True,
                text=True,
            )

            if check_proc.returncode != 0:
                # Commit doesn't exist locally, try to fetch it
                if verbose:
                    print(
                        f"..Fetching {sha[:8]} from origin",
                        file=sys.stderr,
                    )

                fetch_commit_proc = subprocess.run(
                    ["git", "--git-dir", str(bare_repo_dir), "fetch", "origin", sha],
                    capture_output=True,
                    text=True,
                )

                if fetch_commit_proc.returncode != 0:
                    if not silent:
                        print(
                            f"..Cannot fetch {sha[:8]}: {fetch_commit_proc.stderr.strip()}",
                            file=sys.stderr,
                        )
                    continue

            # Create worktree at this SHA
            cmd = [
                "git",
                "--git-dir",
                str(bare_repo_dir),
                "worktree",
                "add",
                str(worktree_dir),
                sha,
            ]

            if verbose:
                print(
                    f"..Creating worktree for {sha[:8]} at {worktree_dir}",
                    file=sys.stderr,
                )

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                if not silent:
                    print(
                        f"..Worktree creation failed for {sha[:8]}: {proc.stderr.strip()}",
                        file=sys.stderr,
                    )
                continue
        else:
            if verbose:
                print(f"..Using cached worktree at {worktree_dir}", file=sys.stderr)

        local_repos.append((repo_full_name, sha, worktree_dir))

    return local_repos


def clone_ehrql_repos(
    repos: list[str],
    cache_dir: pathlib.Path,
    silent: bool = False,
    verbose: bool = False,
    csv_path: pathlib.Path | None = None,
):
    """Clone ehrQL repos from GitHub API search, with optional additional SHAs from CSV.

    Priority:
    1. Always clone latest commit from GitHub API search for each repo
    2. If CSV provided, also clone any additional SHAs for repos found in GitHub search
    3. Report (but don't clone) repos that are in CSV but not in GitHub search

    Args:
        repos: Optional list of specific repos to include
        cache_dir: Directory for caching cloned repos
        silent: Suppress output
        verbose: Verbose output
        csv_path: Optional path to jobs CSV file for additional SHAs

    Returns:
        List of tuples (repo_full_name, sha, local_path) for each repo/SHA combination
    """
    # Always start with GitHub API search
    project_yaml_files = project_yaml_search(verbose=verbose)
    ehrql_repos = group_items_by_repo(project_yaml_files)

    if not silent:
        print(
            f"Found {len(project_yaml_files)} project.yaml files in {len(ehrql_repos)} opensafely ehrql repos",
            file=sys.stderr,
        )

    # Start with GitHub API results (latest commit for each repo)
    all_repos_shas = list(ehrql_repos.items())

    # If CSV provided, add additional SHAs for repos in the GitHub search
    if csv_path:
        csv_repo_shas = parse_jobs_csv(csv_path)

        if not silent:
            total_csv_repos = len(csv_repo_shas)
            total_csv_shas = sum(len(shas) for shas in csv_repo_shas.values())
            print(
                f"Found {total_csv_shas} unique repo/SHA combinations across {total_csv_repos} repos in CSV",
                file=sys.stderr,
            )

        # Track repos in CSV but not in GitHub search
        csv_only_repos = []

        # Add additional SHAs from CSV for repos that are in the GitHub search
        added_shas_count = 0
        for repo_name, csv_shas in csv_repo_shas.items():
            if repo_name in ehrql_repos:
                # This repo is in GitHub search - add any additional SHAs from CSV
                github_sha = ehrql_repos[repo_name]
                for csv_sha in csv_shas:
                    if csv_sha != github_sha:
                        all_repos_shas.append((repo_name, csv_sha))
                        added_shas_count += 1
            else:
                # This repo is in CSV but not in GitHub search
                csv_only_repos.append((repo_name, len(csv_shas)))

        if not silent and added_shas_count > 0:
            print(
                f"Added {added_shas_count} additional SHAs from CSV for repos in GitHub search",
                file=sys.stderr,
            )

        # Report repos in CSV but not in GitHub search
        if csv_only_repos and not silent:
            print("\n" + "=" * 80, file=sys.stderr)
            print("REPOS IN CSV BUT NOT IN GITHUB SEARCH", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print(
                f"\nTotal: {len(csv_only_repos)} repos in CSV were not found in GitHub ehrQL search",
                file=sys.stderr,
            )
            print("(These will NOT be cloned or processed)", file=sys.stderr)
            print("\nRepos:", file=sys.stderr)
            print("-" * 80, file=sys.stderr)
            print_limit = 10
            for repo, sha_count in sorted(csv_only_repos)[:print_limit]:
                print(f"  {repo} ({sha_count} SHA(s) in CSV)", file=sys.stderr)
            if len(csv_only_repos) > print_limit:
                print(
                    f"\n  ... and {len(csv_only_repos) - print_limit} more",
                    file=sys.stderr,
                )
            print("=" * 80 + "\n", file=sys.stderr)

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Cloning ehrQL repos")
    local_ehrql_repos = clone_repos(
        all_repos_shas, repos, cache_dir, silent=silent, verbose=verbose
    )
    return local_ehrql_repos


def get_dataset_files(
    local_repos: list[tuple[str, str, pathlib.Path]],
    silent: bool = False,
    verbose: bool = False,
) -> dict[str, (str, list[str], pathlib.Path)]:
    """Get dataset definition files from local cloned repos.

    Note: When multiple SHAs exist for the same repo, we use a composite key
    "repo@sha" to distinguish them.
    """
    all_dataset_files: dict[str, (str, list[str], pathlib.Path)] = {}
    for repo_full, head_sha, repo_local_dir in local_repos:
        dataset_files = parse_project_yaml(repo_local_dir)

        if dataset_files:
            # Use composite key "repo@sha" to support multiple SHAs per repo
            composite_key = f"{repo_full}@{head_sha}"
            all_dataset_files[composite_key] = (
                head_sha,
                dataset_files,
                repo_local_dir,
            )
            if verbose:
                print(
                    f"..Found {len(dataset_files)} dataset files in project.yaml: {dataset_files}",
                    file=sys.stderr,
                )
        else:
            if verbose:
                print(
                    "..No ehrql generate_dataset commands found in project.yaml",
                    file=sys.stderr,
                )
    return all_dataset_files
