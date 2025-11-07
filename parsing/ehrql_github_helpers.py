import base64
import json
import pathlib
import re
import subprocess
import sys

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
                parts = filtered_command.split()

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
