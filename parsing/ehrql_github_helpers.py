import base64
import csv
import hashlib
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

# Global cache for project.yaml parsing
# Maps content hash -> list of dataset files
_PROJECT_YAML_CACHE: dict[str, list[str]] = {}
_PROJECT_YAML_CACHE_FILE = pathlib.Path(".project_yaml_cache.json")
_PROJECT_YAML_CACHE_MODIFIED = False


class GitHubError(RuntimeError):
    """Generic error for GitHub CLI interactions."""

    pass


def _load_project_yaml_cache() -> None:
    """Load the project.yaml parsing cache from disk."""
    global _PROJECT_YAML_CACHE, _PROJECT_YAML_CACHE_MODIFIED

    if _PROJECT_YAML_CACHE_FILE.exists():
        try:
            with open(_PROJECT_YAML_CACHE_FILE, encoding="utf-8") as f:
                _PROJECT_YAML_CACHE = json.load(f)
        except (json.JSONDecodeError, OSError):
            # If cache is corrupted, start fresh
            _PROJECT_YAML_CACHE = {}
    _PROJECT_YAML_CACHE_MODIFIED = False


def _save_project_yaml_cache() -> None:
    """Save the project.yaml parsing cache to disk if modified."""
    global _PROJECT_YAML_CACHE_MODIFIED

    if _PROJECT_YAML_CACHE_MODIFIED:
        try:
            with open(_PROJECT_YAML_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(_PROJECT_YAML_CACHE, f)
            _PROJECT_YAML_CACHE_MODIFIED = False
        except OSError:
            # Ignore save errors - cache is just an optimization
            pass


def _clear_project_yaml_cache() -> None:
    """Clear the project.yaml parsing cache."""
    global _PROJECT_YAML_CACHE, _PROJECT_YAML_CACHE_MODIFIED

    _PROJECT_YAML_CACHE = {}
    _PROJECT_YAML_CACHE_MODIFIED = False

    if _PROJECT_YAML_CACHE_FILE.exists():
        try:
            _PROJECT_YAML_CACHE_FILE.unlink()
        except OSError:
            pass


def _hash_content(content: str) -> str:
    """Create a hash of content for cache lookup."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


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
    Uses caching based on file content hash to speed up repeated parsing.
    """
    global _PROJECT_YAML_CACHE, _PROJECT_YAML_CACHE_MODIFIED

    try:
        content = (repo_root / "project.yaml").read_text(encoding="utf-8")

        # Check cache first
        content_hash = _hash_content(content)
        if content_hash in _PROJECT_YAML_CACHE:
            return _PROJECT_YAML_CACHE[content_hash]

        # Cache miss - do the parsing
        data = yaml.safe_load(content)

        dataset_files = set()
        if not data:
            result = list(dataset_files)
            _PROJECT_YAML_CACHE[content_hash] = result
            _PROJECT_YAML_CACHE_MODIFIED = True
            return result

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

        result = list(dataset_files)
        # Update cache
        _PROJECT_YAML_CACHE[content_hash] = result
        _PROJECT_YAML_CACHE_MODIFIED = True
        return result
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


def get_all_repos_with_shas_graphql(
    repo_filter: list[str] | None = None,
) -> dict[str, str]:
    """Fetch repos in opensafely org with their HEAD SHAs using GraphQL with pagination.

    This is much faster than REST API as it gets all repos and SHAs in ~4 calls
    instead of 1+n where n is the number of repos. This is because there is no single
    call to the REST API that gets all the SHAs. The fastest is one paginated query to
    get all the repos in an org, then individual calls for each repo to get the default
    branch SHA.

    Args:
        repo_filter: Optional list of repo names to fetch (e.g., ['repo1', 'repo2']).
                    If provided, only these repos are fetched (much faster).
                    If None, all repos in the org are fetched.

    Returns dict of {repo_name: sha} e.g. {'covid-project': 'abc123...'}
    """
    repos_with_shas: dict[str, str] = {}

    if repo_filter:
        # Fast path: fetch specific repos by name
        # Build query to fetch each repo individually
        repo_queries = []
        for i, repo_name in enumerate(repo_filter):
            alias = f"repo{i}"
            repo_queries.append(
                f'{alias}: repository(owner: "opensafely", name: "{repo_name.replace("opensafely/", "")}") {{ defaultBranchRef {{ target {{ ... on Commit {{ oid }} }} }} }}'
            )

        query = "query { " + " ".join(repo_queries) + " }"

        try:
            print(
                f"Fetching {len(repo_filter)} specific repos via GraphQL...",
                file=sys.stderr,
            )
            result = subprocess.run(
                ["gh", "api", "graphql", "-f", f"query={query}"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            data = json.loads(result.stdout)

            if "errors" in data:
                raise GitHubError(f"GraphQL error: {data['errors']}")

            for i, repo_name in enumerate(repo_filter):
                alias = f"repo{i}"
                repo_data = data.get("data", {}).get(alias)
                if (
                    repo_data
                    and repo_data.get("defaultBranchRef")
                    and repo_data["defaultBranchRef"].get("target")
                ):
                    oid = repo_data["defaultBranchRef"]["target"]["oid"]
                    repos_with_shas[repo_name.replace("opensafely/", "")] = oid
                else:
                    if repo_data is None:
                        print(
                            f"  Warning: Repo '{repo_name}' not found in opensafely org",
                            file=sys.stderr,
                        )

            print(f"  Got {len(repos_with_shas)} repos", file=sys.stderr)
        except Exception as e:
            raise GitHubError(f"Failed to fetch specific repos via GraphQL: {e}") from e
    else:
        # Slow path: fetch all repos with pagination
        after_cursor = None

        query = """
query($org: String!, $first: Int!, $after: String) {
  organization(login: $org) {
    repositories(first: $first, after: $after, privacy: PUBLIC) {
      nodes {
        name
        defaultBranchRef {
          target {
            ... on Commit {
              oid
            }
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
"""

        while True:
            cmd = [
                "gh",
                "api",
                "graphql",
                "-f",
                f"query={query}",
                "-f",
                "org=opensafely",
                "-F",
                "first=100",
            ]

            if after_cursor:
                cmd.extend(["-f", f"after={after_cursor}"])

            try:
                print("Fetching 100 repos via GraphQL...", file=sys.stderr)
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30, check=False
                )
                data = json.loads(result.stdout)
                print(
                    f"Received data for {len(data.get('data', {}).get('organization', {}).get('repositories', {}).get('nodes', []))} repos",
                    file=sys.stderr,
                )
                if "errors" in data:
                    raise GitHubError(f"GraphQL error: {data['errors']}")

                repos = data["data"]["organization"]["repositories"]
                for node in repos["nodes"]:
                    if node.get("defaultBranchRef") and node["defaultBranchRef"].get(
                        "target"
                    ):
                        oid = node["defaultBranchRef"]["target"]["oid"]
                        repos_with_shas[node["name"]] = oid

                if not repos["pageInfo"]["hasNextPage"]:
                    break

                after_cursor = repos["pageInfo"]["endCursor"]
            except Exception as e:
                raise GitHubError(f"Failed to fetch repos via GraphQL: {e}") from e

    return repos_with_shas


def project_yaml_search(
    verbose: bool = False, repos: list[str] | None = None
) -> list[dict]:
    """Search for project.yaml files with ehrql content.

    This approach:
    1. Gets all opensafely org repos and their HEAD SHAs via GraphQL (or specific repos if provided)
    2. Checks cache index first (ehrql_repos_index.json)
    3. For uncached repos, shallow clones them to .ehrql_repo_cache/{repo_name}-{sha}/
    4. Searches locally for project.yaml files containing 'ehrql'

    Args:
        verbose: Print detailed logging
        repos: Optional list of specific repo names to search (e.g., ['repo1', 'repo2']).
               If None, searches all repos in opensafely org.

    Returns list of dicts with 'repo_full_name' and 'sha' keys.
    """
    cache_dir = pathlib.Path(".ehrql_repo_cache")
    cache_dir.mkdir(exist_ok=True)

    # Fast cache: stores {repo_name: {"has_ehrql": bool, "sha": str, "no_project_yaml": bool}}
    cache_index_file = cache_dir / "ehrql_repos_index.json"
    cache_index: dict = {}
    if cache_index_file.exists():
        try:
            cache_index = json.load(cache_index_file.open())
        except Exception:
            cache_index = {}

    if verbose:
        print("Fetching all repos and HEAD SHAs via GraphQL...", file=sys.stderr)

    # Use GraphQL to get all repos with SHAs in ~4 calls instead of 300+
    # Pass repo_filter if specific repos were requested
    repos_with_shas = get_all_repos_with_shas_graphql(repo_filter=repos)

    if verbose:
        print(f"Found {len(repos_with_shas)} repos in opensafely org", file=sys.stderr)

    items: list[dict] = []
    cloned_count = 0
    cached_count = 0
    no_project_yaml_count = 0
    index_hits = 0
    cache_updated = False

    # For each repo, check cache index first, then clone if needed
    for i, repo_name in enumerate(repos_with_shas.keys()):
        repo_full_name = f"opensafely/{repo_name}"
        head_sha = repos_with_shas[repo_name]

        if (i + 1) % 20 == 0:
            print(
                f"  Checked {i + 1}/{len(repos_with_shas)} repos (index hits: {index_hits}, cached: {cached_count}, cloned: {cloned_count}, no project.yaml: {no_project_yaml_count}, ehrql found: {len(items)})",
                file=sys.stderr,
            )

        try:
            # Fast path: check index first
            if repo_name in cache_index:
                entry = cache_index[repo_name]
                index_hits += 1

                if entry.get("no_project_yaml"):
                    no_project_yaml_count += 1
                    continue

                if entry.get("has_ehrql"):
                    items.append(
                        {
                            "repo_full_name": repo_full_name,
                            "sha": entry.get("sha", ""),
                        }
                    )
                    cached_count += 1
                    continue
                else:
                    # In index but no ehrql
                    continue

            # Use the same cache naming as clone_ehrql_repos: {repo_name}-{first_8_chars_of_sha}
            cache_key = f"{repo_name}-{head_sha[:8]}"
            repo_cache_dir = cache_dir / cache_key

            # Check if we already have project.yaml for this SHA
            project_yaml_path = repo_cache_dir / "project.yaml"

            if not project_yaml_path.exists():
                # Check if the directory exists (might be cloned but no project.yaml)
                if repo_cache_dir.exists():
                    no_project_yaml_count += 1
                    if verbose:
                        print(
                            f"  {repo_full_name}: cached but no project.yaml",
                            file=sys.stderr,
                        )
                    continue

                # Need to clone
                cloned_count += 1
                print(f"  Cloning {repo_full_name}@{head_sha[:8]}...", file=sys.stderr)

                # Full shallow clone (with working tree)
                result = subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        f"https://github.com/{repo_full_name}.git",
                        str(repo_cache_dir),
                    ],
                    capture_output=True,
                    check=False,
                    timeout=30,
                )

                if result.returncode != 0:
                    print(
                        f"  Failed to clone {repo_full_name}: {result.stderr.decode()[:100]}",
                        file=sys.stderr,
                    )
                    continue

                # After cloning, check again for project.yaml
                if not project_yaml_path.exists():
                    no_project_yaml_count += 1
                    if verbose:
                        print(
                            f"  {repo_full_name}: cloned but no project.yaml",
                            file=sys.stderr,
                        )
                    continue
            else:
                cached_count += 1

            # Check for project.yaml with ehrql content
            if project_yaml_path.exists():
                try:
                    content = project_yaml_path.read_text(encoding="utf-8")
                    if "ehrql" in content.lower():
                        items.append(
                            {
                                "repo_full_name": repo_full_name,
                                "sha": head_sha,
                            }
                        )

                        # Update index
                        cache_index[repo_name] = {
                            "has_ehrql": True,
                            "sha": head_sha,
                            "no_project_yaml": False,
                        }
                        cache_updated = True

                        if verbose:
                            print(f"  Found ehrql in {repo_full_name}", file=sys.stderr)
                    else:
                        # Has project.yaml but no ehrql
                        cache_index[repo_name] = {
                            "has_ehrql": False,
                            "sha": head_sha,
                            "no_project_yaml": False,
                        }
                        cache_updated = True
                except Exception as e:
                    if verbose:
                        print(
                            f"  Error reading {project_yaml_path}: {e}", file=sys.stderr
                        )
            else:
                # No project.yaml
                cache_index[repo_name] = {
                    "has_ehrql": False,
                    "sha": head_sha,
                    "no_project_yaml": True,
                }
                cache_updated = True
                if verbose:
                    print(
                        f"  No project.yaml found in {repo_full_name}", file=sys.stderr
                    )
        except subprocess.TimeoutExpired:
            if verbose:
                print(f"  Timeout checking {repo_full_name}", file=sys.stderr)
        except Exception as e:
            if verbose:
                print(f"  Error processing {repo_full_name}: {e}", file=sys.stderr)

    # Save updated cache index
    if cache_updated:
        try:
            json.dump(cache_index, cache_index_file.open("w"), indent=2)
        except Exception:
            pass

    return items


def group_items_by_repo(items: list[dict]) -> dict:
    grouped: dict = {}
    for it in items:
        repo = it.get("repo_full_name")
        head_sha = it.get("sha")
        if not repo or not head_sha:
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
    # Pass the repos parameter to only search specific repos if provided
    project_yaml_files = project_yaml_search(
        verbose=verbose, repos=repos if repos else None
    )
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
    force: bool = False,
) -> dict[str, (str, list[str], pathlib.Path)]:
    """Get dataset definition files from local cloned repos.

    Note: When multiple SHAs exist for the same repo, we use a composite key
    "repo@sha" to distinguish them.

    Args:
        local_repos: List of (repo_full_name, sha, local_path) tuples
        silent: Suppress output
        verbose: Verbose output
        force: If True, clear cache and force re-parsing
    """
    # Load cache on first call, or clear it if force=True
    if force:
        _clear_project_yaml_cache()
    elif not _PROJECT_YAML_CACHE:
        _load_project_yaml_cache()

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

    # Save cache if modified
    _save_project_yaml_cache()

    return all_dataset_files
