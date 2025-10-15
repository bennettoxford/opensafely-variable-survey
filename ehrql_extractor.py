"""Collect variable definitions from ehrql dataset definition files across a GitHub org.

See README.md for usage.
"""

from __future__ import annotations

import argparse
import ast
import base64
import builtins
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd
import pyarrow as _pa
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as _pa_ipc
import yaml
from ehrql.query_model import nodes as qm


def convert_spoofed_data(verbose: bool = False) -> int:
    root_dir = pathlib.Path(__file__).parent
    data_dir = root_dir / "spoofed_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    input_path = data_dir / "csv_data.csv"

    if verbose:
        print("Creating a .csv.gz and .arrow file from spoofed_data.csv...")
        print(f"Reading CSV from {input_path}")
    df = pd.read_csv(input_path)

    # Write out compressed CSV file
    gz_path = data_dir / f"{input_path.stem}.csv.gz"
    if verbose:
        print(f"Writing gzipped CSV to {gz_path}")
    df.to_csv(gz_path, index=False, compression="gzip")

    # Write out Arrow IPC file
    arrow_path = data_dir / f"{input_path.stem}.arrow"
    if verbose:
        print(f"Writing Arrow IPC file to {arrow_path}")
    try:
        # Simple heuristic: if the first non-null value of a column matches YYYY-MM-DD
        # then treat the whole column as date and convert to datetime.
        date_cols = set()
        import re

        date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for col in df.columns:
            # Skip numeric columns
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                continue
            # Find first value
            val = df[col].tolist()[0]
            if date_re.match(val):
                date_cols.add(col)
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

        # Build pyarrow arrays, converting detected date columns to date32
        pa_arrays = {}
        for col in df.columns:
            if col in date_cols:
                # Already converted to python date objects in df[col]
                py_dates = [d if pd.notna(d) else None for d in df[col]]
                pa_arrays[col] = pa.array(py_dates, type=pa.date32())
            else:
                # Let pyarrow infer type from the pandas column
                pa_arrays[col] = pa.array(df[col].tolist())

        table = pa.table(pa_arrays)
        feather.write_feather(table, arrow_path)
    except Exception as e:  # pragma: no cover - surface errors to user
        print(f"Failed to write Arrow file: {e}", file=sys.stderr)
        return 3

    if verbose:
        print("Conversion complete")
    return 0


class GitHubError(RuntimeError):
    """Generic error for GitHub CLI interactions."""

    pass


GH_API_HEADERS = [
    "Accept: application/vnd.github+json",
    "X-GitHub-Api-Version: 2022-11-28",
]


@contextmanager
def working_directory(path):
    """Context manager for changing the working directory"""
    prev_cwd = pathlib.Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def run_gh(args: list[str], expect_json: bool = True) -> dict | list | str:
    """Run a gh CLI command and return parsed JSON or raw stdout.

    Raises GitHubError on non‑zero exit or JSON parse failure.
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


def code_search(verbose: bool = False) -> list[dict]:
    """Return list of code search result items for the whole org.

    Uses paginated search up to 1000 results (GitHub API inherent cap). Each item includes
    repository info and path. We add a derived key 'repo_full_name' for convenience.
    """
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


def fetch_file_content(owner: str, repo: str, path: str, ref: str) -> tuple[str, str]:
    """Return (decoded_text, blob_sha) for file path at commit ref using gh api contents."""
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


@dataclass
class VariableRecord:
    project_name: str
    project_sha: str
    file_name: str
    variable_name: str
    series_type: str
    line_no: int
    qm_node: str


def setup_spoofs(silent: bool = False, verbose: bool = False) -> None:
    # Grab the ehrql module so we can spoof bits of it
    global ehrql_mod
    import ehrql as ehrql_mod

    if ehrql_mod is None:
        if not silent:
            print("..ehrql not importable; cannot setup spoofs", file=sys.stderr)
            raise RuntimeError("ehrql import failed")

    import importlib as _importlib

    # Module aliasing for backward compatibility
    # 1. ehrql.tables.beta.* -> ehrql.tables.*
    #    (beta namespace is deprecated, redirect to current tables)
    sys.modules["ehrql.tables.beta"] = _importlib.import_module("ehrql.tables")
    sys.modules["ehrql.tables.beta.tpp"] = _importlib.import_module("ehrql.tables.tpp")

    # 2. databuilder.* -> ehrql.*
    #    (databuilder was the old name for ehrql)
    sys.modules["databuilder"] = ehrql_mod
    sys.modules["databuilder.ehrql"] = (
        ehrql_mod  # Special case: databuilder.ehrql -> ehrql
    )
    sys.modules["databuilder.codes"] = _importlib.import_module("ehrql.codes")
    sys.modules["databuilder.tables"] = _importlib.import_module("ehrql.tables")
    sys.modules["databuilder.tables.tpp"] = _importlib.import_module("ehrql.tables.tpp")
    sys.modules["databuilder.tables.beta"] = _importlib.import_module("ehrql.tables")
    sys.modules["databuilder.tables.beta.tpp"] = _importlib.import_module(
        "ehrql.tables.tpp"
    )

    # Spoof renamed table: hospital_admissions -> apcs
    # When code tries to import hospital_admissions, redirect to apcs
    tpp_module = _importlib.import_module("ehrql.tables.tpp")

    original_apcs = tpp_module.apcs

    # Create alias: hospital_admissions -> apcs
    tpp_module.hospital_admissions = original_apcs

    # Add primary_diagnoses as a property on the apcs CLASS (not instance)
    # so it's available on all instances including those created by where(), sort_by(), etc.
    apcs_class = original_apcs.__class__

    # Create a property that redirects primary_diagnoses to primary_diagnosis
    def primary_diagnoses_property(self):
        if verbose:
            print(
                "..Redirecting apcs.primary_diagnoses to apcs.primary_diagnosis",
                file=sys.stderr,
            )
        return self.primary_diagnosis

    # Add the property to the class
    setattr(
        apcs_class,
        "primary_diagnoses",
        property(primary_diagnoses_property),
    )

    if verbose:
        print(
            "..Created hospital_admissions alias and primary_diagnoses property on apcs class",
            file=sys.stderr,
        )

    # Spoof ons_deaths.where() to just return self
    # ons_deaths changed from EventFrame to PatientFrame, so where() is no longer valid
    # Old code may still call where() on it, so we make it a no-op
    ons_deaths = tpp_module.ons_deaths
    ons_deaths_class = ons_deaths.__class__

    # Create a method that just returns self
    def spoofed_event_frame_method(self, condition=None):
        if verbose:
            print(
                "..Intercepting event frame method on ons_deaths - returning self (ons_deaths is now a PatientFrame)",
                file=sys.stderr,
            )
        return self

    # Add the spoofed where method to the class
    setattr(ons_deaths_class, "where", spoofed_event_frame_method)
    setattr(ons_deaths_class, "sort_by", spoofed_event_frame_method)
    setattr(ons_deaths_class, "first_for_patient", spoofed_event_frame_method)
    setattr(ons_deaths_class, "last_for_patient", spoofed_event_frame_method)

    if verbose:
        print(
            "..Added no-op where() method to ons_deaths class",
            file=sys.stderr,
        )

    repo_root = pathlib.Path(__file__).parent  # Root of this analysis repo
    spoof_dir = repo_root / "spoofed_data"

    # Spoof args for when the script manually process system args
    global _spoofed_args
    spoofed_args_path = spoof_dir / "args.json"
    with open(spoofed_args_path, encoding="utf-8") as f:
        _spoofed_args = json.load(f)

    # Spoof parameters for when the user calls ehrql.get_parameter()
    global _spoofed_params
    spoofed_params_path = spoof_dir / "parameters.json"

    with open(spoofed_params_path, encoding="utf-8") as _f:
        _spoofed_params = json.load(_f)

    def spoofed_get_parameter(*a, **kw):
        default = kw.get("default")
        if default is not None:
            return default
        # Accept either name=... or positional name
        name = kw.get("name") if isinstance(kw.get("name"), str) else a[0]
        if not silent:
            print(f"..Spoofing get_parameter({name})", file=sys.stderr)
        if name and name in _spoofed_params:
            if repo_name in _spoofed_params[name]:
                return _spoofed_params[name][repo_name]
            return _spoofed_params[name]["default"]
        return None

    setattr(ehrql_mod, "get_parameter", spoofed_get_parameter)

    # Spoof case() function to redirect "default" kwarg to "otherwise"
    original_case = getattr(ehrql_mod, "case", None)

    def spoofed_case(*args, **kwargs):
        # If "default" kwarg is present, rename it to "otherwise"
        if "default" in kwargs:
            if verbose:
                print(
                    "..Redirecting case(default=...) to case(otherwise=...)",
                    file=sys.stderr,
                )
            kwargs["otherwise"] = kwargs.pop("default")
        return original_case(*args, **kwargs)

    setattr(ehrql_mod, "case", spoofed_case)

    # Spoof argparse.ArgumentParser.add_argument
    original_argparse_add = argparse.ArgumentParser.add_argument

    def _spoofed_add_argument(self, *a, **kw):
        for opt in a:
            if isinstance(opt, str):
                # Ignore the default help options added by argparse
                if opt in ("-h", "--help"):
                    continue
                # If the declared option string isn't in argv or our spoofed set,
                # warn the user so they can add it to spoofed_args.json.
                if (
                    opt not in sys.argv
                    and repo_name not in _spoofed_args
                    or opt not in _spoofed_args[repo_name]
                ):
                    if verbose:
                        print(
                            f"  CLI option declared {opt!r} not present in sys.argv; consider adding it to spoofed_data/args.json",
                            file=sys.stderr,
                        )
        return original_argparse_add(self, *a, **kw)

    argparse.ArgumentParser.add_argument = _spoofed_add_argument

    # Monkey-patch open() to redirect file reads to spoofed files
    # This allows dataset definitions to load study-specific JSON, csv files etc (e.g.,
    # study_dates.json) that aren't committed to the repository but are generated
    # during study execution.
    builtin_open = open
    convert_spoofed_data(
        verbose=verbose
    )  # ensure csv.gz and .arrow spoofed files are generated
    spoofed_json_path = spoof_dir / "json_data.json"
    spoofed_csv_path = spoof_dir / "csv_data.csv"
    spoofed_csv_gz_path = spoof_dir / "csv_data.csv.gz"
    spoofed_arrow_path = spoof_dir / "csv_data.arrow"

    def spoofed_open(file, mode="r", *args, **kwargs):
        # Intercept reads of JSON files and redirect to spoofed file
        if "r" in mode and isinstance(file, (str, pathlib.Path)):
            file_str = str(file)
            if file_str.endswith(".json"):
                if verbose:
                    print(
                        f"..Redirecting JSON read from {file_str} to {spoofed_json_path}",
                        file=sys.stderr,
                    )
                return builtin_open(spoofed_json_path, mode, *args, **kwargs)
            elif file_str.endswith(".csv"):
                if verbose:
                    print(
                        f"..Redirecting CSV read from {file_str} to {spoofed_csv_path}",
                        file=sys.stderr,
                    )
                return builtin_open(spoofed_csv_path, mode, *args, **kwargs)
            elif file_str.endswith(".csv.gz"):
                if verbose:
                    print(
                        f"..Redirecting CSV.gz read from {file_str} to {spoofed_csv_gz_path}",
                        file=sys.stderr,
                    )
                return builtin_open(spoofed_csv_gz_path, mode, *args, **kwargs)
            elif file_str.endswith(".arrow"):
                if verbose:
                    print(
                        f"..Redirecting Arrow read from {file_str} to {spoofed_arrow_path}",
                        file=sys.stderr,
                    )
                return builtin_open(spoofed_arrow_path, mode, *args, **kwargs)
        return builtin_open(file, mode, *args, **kwargs)

    # Apply the monkey patch
    builtins.open = spoofed_open

    # Prepare to optionally monkey-patch pyarrow to redirect memory_map/open_file
    original_pyarrow_memory_map = None
    original_pyarrow_ipc_open_file = None

    # Save originals
    original_pyarrow_memory_map = _pa.memory_map
    original_pyarrow_ipc_open_file = _pa_ipc.open_file

    def _spoofed_memory_map(path, mode="rb", *a, **kw):
        p = str(path)
        # Redirect arrow reads to our spoofed arrow payload
        if p.endswith(".arrow") or p.endswith(".feather"):
            if verbose:
                print(
                    f"..Redirecting pyarrow.memory_map from {p} to {spoofed_arrow_path}",
                    file=sys.stderr,
                )
            return original_pyarrow_memory_map(str(spoofed_arrow_path), mode, *a, **kw)
        return original_pyarrow_memory_map(path, mode, *a, **kw)

    def _spoofed_ipc_open_file(source, *a, **kw):
        # source may be a path-like or a MemoryMappedFile; try to coerce to str
        try:
            s = str(source)
        except Exception:
            s = ""
        if s.endswith(".arrow") or s.endswith(".feather"):
            if verbose:
                print(
                    f"..Redirecting pyarrow.ipc.open_file from {s} to {spoofed_arrow_path}",
                    file=sys.stderr,
                )
            mm = original_pyarrow_memory_map(str(spoofed_arrow_path), "rb")
            return original_pyarrow_ipc_open_file(mm, *a, **kw)
        return original_pyarrow_ipc_open_file(source, *a, **kw)

    # Apply patches
    _pa.memory_map = _spoofed_memory_map
    _pa_ipc.open_file = _spoofed_ipc_open_file

    # Monkey-patch pathlib.Path.is_file() to return True for CSV/CSV.gz files
    # This allows dataset definitions to check for file existence without errors
    # This works for both Path and PosixPath since PosixPath inherits from Path
    original_is_file = pathlib.Path.is_file

    def spoofed_is_file(self):
        # Check if it's a CSV or CSV.gz file that we want to spoof
        path_str = str(self)
        if (
            path_str.endswith(".csv")
            or path_str.endswith(".csv.gz")
            or path_str.endswith(".arrow")
        ):
            if verbose:
                print(
                    f"..Spoofing is_file() check for {path_str} -> True",
                    file=sys.stderr,
                )
            return True
        # Otherwise use the original method
        # Call it as a bound method to avoid issues with non-existent files
        try:
            return original_is_file(self)
        except (OSError, FileNotFoundError):
            # If the file doesn't exist, return False instead of raising
            return False

    pathlib.Path.is_file = spoofed_is_file

    # Monkey-patch Dataset class to redirect configure_dummy_dataset to configure_dummy_data
    # This handles cases where older code uses the deprecated method name
    Dataset = getattr(ehrql_mod, "Dataset", None)
    original_configure_dummy_data = getattr(Dataset, "configure_dummy_data", None)

    def spoofed_configure_dummy_dataset(self, *args, **kwargs):
        if verbose:
            print(
                "..Redirecting configure_dummy_dataset to configure_dummy_data",
                file=sys.stderr,
            )
        return original_configure_dummy_data(self, *args, **kwargs)

    Dataset.configure_dummy_dataset = spoofed_configure_dummy_dataset

    original_define_population = getattr(Dataset, "define_population", None)

    def spoofed_define_population(self, *args, **kwargs):
        if verbose:
            print("..Calling define_population", file=sys.stderr)
        if hasattr(self, "population"):
            del self.population
            if verbose:
                print(
                    "Second call to define_population ignored as this is now an error",
                    file=sys.stderr,
                )
        return original_define_population(self, *args, **kwargs)

    Dataset.define_population = spoofed_define_population

    BaseFrame = ehrql_mod.query_language.BaseFrame
    original_select_column = BaseFrame._select_column

    def _spoofed_select_column(self, name: str):
        # Redirect old column name to new name
        if name == "primary_diagnoses":
            if verbose:
                print(
                    "..Redirecting select_column('primary_diagnoses') to 'primary_diagnosis'",
                    file=sys.stderr,
                )
            name = "primary_diagnosis"
        return original_select_column(self, name)

    BaseFrame._select_column = _spoofed_select_column

    # Spoof rename of if_null_then to when_null_then
    # Create a wrapper that properly delegates to when_null_then
    BaseSeries = ehrql_mod.query_language.BaseSeries
    original_when_null_then = BaseSeries.when_null_then

    # Create a proper method that will work for all subclasses
    def if_null_then(self, *args, **kwargs):
        return original_when_null_then(self, *args, **kwargs)

    BaseSeries.if_null_then = if_null_then

    if verbose:
        print(
            "..Created if_null_then alias pointing to when_null_then",
            file=sys.stderr,
        )


def spoof_args(verbose: bool = False) -> list:
    if repo_name in _spoofed_args:
        if verbose:
            print(f"..Using spoofed args for {repo_name}: {_spoofed_args[repo_name]}")
        return _spoofed_args[repo_name]

    if verbose:
        print(f"..No spoofed args for {repo_name}")

    return []


def extract_variable_line_numbers(file_path: pathlib.Path) -> dict[str, int]:
    """Parse Python file with AST to extract line numbers for dataset variable assignments.

    Handles:
    - dataset.variable_name = expression
    - dataset.define(variable_name=expression)
    - dataset.add_column(variable_name, expression)
    - dataset = generate_dataset() (looks inside generate_dataset function if in same file or imported)

    Returns dict mapping variable_name -> line_number
    """
    line_numbers = {}
    line_number_regexes = []

    try:
        parent_dir = file_path.parent
        with open(file_path, encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=str(file_path))

        # Collect imports and function definitions
        function_defs = {}  # Maps function name to the function node
        imported_modules = {}  # Maps imported name to (module, original_name)

        for node in ast.walk(tree):
            # Collect all function definitions in this file
            if isinstance(node, ast.FunctionDef):
                function_defs[node.name] = node

            # Track imports: from module import function
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                for alias in node.names:
                    imported_name = alias.asname if alias.asname else alias.name
                    imported_modules[imported_name] = (module_name, alias.name)

            # Track imports: import module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_name = alias.asname if alias.asname else alias.name
                    imported_modules[imported_name] = (alias.name, None)

        # Find where dataset is assigned
        for node in ast.walk(tree):
            # Find: dataset = something()
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "dataset":
                        # Check if it's assigned from a function call
                        if isinstance(node.value, ast.Call):
                            # Get the function name
                            func_name = None
                            if isinstance(node.value.func, ast.Name):
                                func_name = node.value.func.id

                            if func_name == "create_dataset":
                                continue
                            # First, try to find the function in this file
                            if func_name and func_name in function_defs:
                                func_def = function_defs[func_name]
                                # Look for dataset.var assignments inside this function
                                for func_node in ast.walk(func_def):
                                    if isinstance(func_node, ast.Assign):
                                        for func_target in func_node.targets:
                                            if isinstance(func_target, ast.Attribute):
                                                if (
                                                    isinstance(
                                                        func_target.value, ast.Name
                                                    )
                                                    and func_target.value.id
                                                    == "dataset"
                                                ):
                                                    var_name = func_target.attr
                                                    line_numbers[var_name] = (
                                                        func_node.lineno
                                                    )

                                    # Also handle dataset.define() inside the function
                                    elif isinstance(func_node, ast.Call):
                                        if isinstance(func_node.func, ast.Attribute):
                                            if (
                                                isinstance(
                                                    func_node.func.value, ast.Name
                                                )
                                                and func_node.func.value.id == "dataset"
                                            ):
                                                # Handle dataset.define(variable_name=expression)
                                                if func_node.func.attr == "define":
                                                    print("DEFINE")
                                                    sys.exit(0)
                                                    for keyword in func_node.keywords:
                                                        if keyword.arg:
                                                            line_numbers[
                                                                keyword.arg
                                                            ] = func_node.lineno
                                                # Handle dataset.add_column("variable_name", expression)
                                                elif (
                                                    func_node.func.attr == "add_column"
                                                ):
                                                    if (
                                                        func_node.args
                                                        and len(func_node.args) >= 1
                                                    ):
                                                        first_arg = func_node.args[0]
                                                        # Extract string literal from first argument
                                                        if isinstance(
                                                            first_arg, ast.Constant
                                                        ):
                                                            var_name = first_arg.value
                                                            if isinstance(
                                                                var_name, str
                                                            ):
                                                                line_numbers[
                                                                    var_name
                                                                ] = func_node.lineno

                            # Second, try to find the function in imported modules
                            elif func_name and func_name in imported_modules:
                                module_name, original_func_name = imported_modules[
                                    func_name
                                ]
                                # Try to resolve the imported module relative to current file
                                if module_name:
                                    module_parts = module_name.split(".")
                                    module_rel_path = pathlib.Path(*module_parts)
                                    # Build candidate files walking up the directory tree so that
                                    # imports like `from analysis.foo import bar` resolve when the
                                    # current file already lives inside analysis/.
                                    candidate_files: list[pathlib.Path] = []
                                    for ancestor in [parent_dir, *parent_dir.parents]:
                                        base = ancestor / module_rel_path
                                        candidate_files.append(base.with_suffix(".py"))
                                        candidate_files.append(base / "__init__.py")
                                        if ancestor == ancestor.parent:
                                            break
                                    # Deduplicate while preserving order
                                    seen_candidates = set()
                                    ordered_candidates = []
                                    for candidate in candidate_files:
                                        candidate_str = str(candidate)
                                        if candidate_str not in seen_candidates:
                                            seen_candidates.add(candidate_str)
                                            ordered_candidates.append(candidate)
                                    for module_file in ordered_candidates:
                                        if not module_file.exists():
                                            continue
                                        try:
                                            with open(
                                                module_file, encoding="utf-8"
                                            ) as mf:
                                                module_source = mf.read()
                                            module_tree = ast.parse(
                                                module_source, filename=str(module_file)
                                            )

                                            target_func_name = (
                                                original_func_name or func_name
                                            )
                                            for module_node in ast.walk(module_tree):
                                                if (
                                                    isinstance(
                                                        module_node, ast.FunctionDef
                                                    )
                                                    and module_node.name
                                                    == target_func_name
                                                ):
                                                    for func_node in ast.walk(
                                                        module_node
                                                    ):
                                                        if isinstance(
                                                            func_node, ast.Assign
                                                        ):
                                                            for (
                                                                func_target
                                                            ) in func_node.targets:
                                                                if isinstance(
                                                                    func_target,
                                                                    ast.Attribute,
                                                                ):
                                                                    if (
                                                                        isinstance(
                                                                            func_target.value,
                                                                            ast.Name,
                                                                        )
                                                                        and func_target.value.id
                                                                        == "dataset"
                                                                    ):
                                                                        var_name = func_target.attr
                                                                        line_numbers[
                                                                            var_name
                                                                        ] = func_node.lineno

                                                        elif isinstance(
                                                            func_node, ast.Call
                                                        ):
                                                            if isinstance(
                                                                func_node.func,
                                                                ast.Attribute,
                                                            ):
                                                                if (
                                                                    isinstance(
                                                                        func_node.func.value,
                                                                        ast.Name,
                                                                    )
                                                                    and func_node.func.value.id
                                                                    == "dataset"
                                                                ):
                                                                    # Handle dataset.define(variable_name=expression)
                                                                    if (
                                                                        func_node.func.attr
                                                                        == "define"
                                                                    ):
                                                                        print("DEFINE")
                                                                        sys.exit(0)
                                                                        for keyword in func_node.keywords:
                                                                            if keyword.arg:
                                                                                line_numbers[
                                                                                    keyword.arg
                                                                                ] = func_node.lineno
                                                                    # Handle dataset.add_column("variable_name", expression)
                                                                    elif (
                                                                        func_node.func.attr
                                                                        == "add_column"
                                                                    ):
                                                                        if (
                                                                            func_node.args
                                                                            and len(
                                                                                func_node.args
                                                                            )
                                                                            >= 1
                                                                        ):
                                                                            first_arg = func_node.args[
                                                                                0
                                                                            ]
                                                                            if isinstance(
                                                                                first_arg,
                                                                                ast.Constant,
                                                                            ):
                                                                                var_name = first_arg.value
                                                                                if isinstance(
                                                                                    var_name,
                                                                                    str,
                                                                                ):
                                                                                    line_numbers[
                                                                                        var_name
                                                                                    ] = func_node.lineno
                                                    break
                                            break
                                        except Exception:
                                            continue

        # Second pass: look for direct dataset assignments at module level
        for node in ast.walk(tree):
            # Handle: dataset.variable_name = expression
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        # Check if it's dataset.something
                        if (
                            isinstance(target.value, ast.Name)
                            and target.value.id == "dataset"
                        ):
                            var_name = target.attr
                            # Only override if we haven't seen this variable yet
                            # (function definitions take precedence as they're where vars are actually defined)
                            if var_name not in line_numbers:
                                line_numbers[var_name] = node.lineno

            # Handle: dataset.define(variable_name=expression, ...) and dataset.add_column()
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "dataset"
                    ):
                        # Handle dataset.define(variable_name=expression)
                        if node.func.attr == "define":
                            print("DEFINE")
                            sys.exit(0)
                            # Extract keyword arguments
                            for keyword in node.keywords:
                                if keyword.arg and keyword.arg not in line_numbers:
                                    line_numbers[keyword.arg] = node.lineno

                        # Handle dataset.add_column("variable_name", expression)
                        elif node.func.attr == "add_column":
                            if node.args and len(node.args) >= 1:
                                first_arg = node.args[0]
                                # Extract string literal from first argument
                                if isinstance(first_arg, ast.Constant):
                                    var_name = first_arg.value
                                    if (
                                        isinstance(var_name, str)
                                        and var_name not in line_numbers
                                    ):
                                        line_numbers[var_name] = node.lineno
                                elif isinstance(first_arg, ast.JoinedStr):
                                    # Handle f-strings by concatenating into a regex-like string
                                    parts = []
                                    for value in first_arg.values:
                                        if isinstance(value, ast.Constant):
                                            parts.append(str(value.value))
                                        else:
                                            parts.append(".+")
                                    var_regex = "".join(parts)
                                    if var_regex not in line_numbers:
                                        line_number_regexes.append(
                                            (var_regex, node.lineno)
                                        )

        return line_numbers, line_number_regexes

    except Exception:
        # If AST parsing fails, return empty dict
        return {}, []


def compact_qm_node(qm_node: qm.Node) -> str:
    # Navigate all dataclass fields of each node recursively
    # When encountering a SelectTable, replace it entirely with the string from the table name
    try:
        if isinstance(qm_node, qm.SelectTable):
            return f"Table({qm_node.name})"
        elif isinstance(qm_node, qm.SelectPatientTable):
            return f"Table({qm_node.name})"
        elif isinstance(qm_node, qm.Node):
            fields = {}
            for field_name in list(qm_node.__dataclass_fields__):
                field_value = getattr(qm_node, field_name)
                if isinstance(field_value, qm.Node):
                    fields[field_name] = compact_qm_node(field_value)
                elif field_name == "cases" and isinstance(field_value, dict):
                    fields[field_name] = ", ".join(
                        [
                            f"if:{compact_qm_node(k) if isinstance(k, qm.Node) else k}->then:{v}"
                            for k, v in field_value.items()
                        ]
                    )
                elif isinstance(field_value, list):
                    fields[field_name] = ", ".join(
                        [
                            compact_qm_node(item) if isinstance(item, qm.Node) else item
                            for item in field_value
                        ]
                    )
                elif isinstance(field_value, datetime.date):
                    fields[field_name] = "{{DATE}}"
                elif isinstance(field_value, str):
                    fields[field_name] = field_value
                elif isinstance(field_value, frozenset):
                    fields[field_name] = field_value
                elif isinstance(field_value, Enum):
                    fields[field_name] = field_value.name
                elif isinstance(field_value, int):
                    fields[field_name] = field_value
                elif isinstance(field_value, float):
                    fields[field_name] = field_value
                else:
                    fields[field_name] = field_value
            field_strs = [f"{k}={v}" for k, v in fields.items()]
            return f"{qm_node.__class__.__name__}({', '.join(field_strs)})"
    except Exception as e:
        print(f"Error compacting QM node: {e}")
        return str(qm_node)


def get_runtime_dataset_variables(
    files: list[str],
    repo_root: pathlib.Path,
    head_sha: str,
    silent: bool = False,
    verbose: bool = False,
) -> set[str]:
    """Execute dataset definition modules to capture runtime variable types.

    Adds StrPatientSeries (etc) as series_type when available.

    Returns a set of dataset file paths that had no captured datasets
    (and whose rows should be excluded from the final output).
    """

    variables: list[VariableRecord] = []

    spoofed_args = spoof_args(verbose)

    for rel_path in files:
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            if verbose:
                print(f"..File {rel_path} does not exist; skipping", file=sys.stderr)
            continue

        abs_path = abs_path.resolve()

        if verbose:
            print(f"..Collecting runtime variables for {abs_path}", file=sys.stderr)

        # Extract line numbers from AST before executing the module
        variable_line_numbers, variable_line_number_regexes = (
            extract_variable_line_numbers(abs_path)
        )
        if verbose and variable_line_numbers:
            print(
                f"....Extracted {len(variable_line_numbers)} variable line numbers from AST",
                f"......and {len(variable_line_number_regexes)} variable line number regexes from AST",
                file=sys.stderr,
            )

        # RESET TO INITIAL STATE BEFORE EACH REPO
        reset_modules_to_snapshot()
        # Ensure repo root importable (for intra-repo relative imports)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        # Ensure the dataset file's own directory is on sys.path so that sibling modules
        # (e.g. codelists.py living beside dataset_definition.py) can be imported using
        # bare module names like `from codelists import X`. Many OpenSafely repos keep
        # analysis scripts in a subfolder (e.g. analysis/). Adding that folder directly
        # mirrors running the script from within that directory.
        parent_dir = abs_path.parent
        if str(parent_dir) in sys.path:
            sys.path.remove(str(parent_dir))
        sys.path.insert(0, str(parent_dir))  # Insert at position 0, not append

        parent_parent_dir = parent_dir.parent
        if str(parent_parent_dir) in sys.path:
            sys.path.remove(str(parent_parent_dir))
        sys.path.insert(1, str(parent_parent_dir))  # Position 1, after parent dir

        # Change working directory to repo root so that codelist_from_csv() calls
        # with relative paths (e.g., "codelists/foo.csv") resolve correctly.
        with working_directory(repo_root):

            def get_trace(e: Exception):
                tb = sys.exc_info()[2]
                # Walk to the last traceback frame
                while tb and tb.tb_next:
                    tb = tb.tb_next
                if tb:
                    frame = tb.tb_frame
                    fname = frame.f_code.co_filename
                    lineno = tb.tb_lineno
                    return f"{fname}:{lineno}"
                return ""

            mod_name = f"ehrql_runtime_{head_sha[:8]}_{rel_path.replace('/', '_').replace('.', '_')}"
            try:
                spec = importlib.util.spec_from_file_location(mod_name, abs_path)
                if spec and spec.loader:  # type: ignore
                    mod = importlib.util.module_from_spec(spec)  # type: ignore
                    sys.modules[mod_name] = mod
                    # Prevent the executed module from seeing our script's argv
                    prev_argv = sys.argv
                    try:
                        # combine file name with spoofed args if they exist
                        sys.argv = [str(abs_path)] + spoofed_args
                        #
                        try:
                            spec.loader.exec_module(mod)  # type: ignore
                            if hasattr(mod, "dataset"):
                                for var_name, series in mod.dataset._variables.items():  # type: ignore[attr-defined]
                                    # Get line number from AST parsing
                                    line_no = variable_line_numbers.get(var_name, -1)

                                    if line_no < 0 and variable_line_number_regexes:
                                        # Try to match variable name against regexes
                                        for (
                                            var_regex,
                                            regex_line_no,
                                        ) in variable_line_number_regexes:
                                            if re.fullmatch(var_regex, var_name):
                                                line_no = regex_line_no
                                                break
                                    variables.append(
                                        VariableRecord(
                                            project_name=repo_name,
                                            project_sha=head_sha,
                                            file_name=rel_path,
                                            variable_name=var_name,
                                            series_type=series.__class__.__name__,
                                            line_no=line_no,
                                            qm_node=compact_qm_node(series._qm_node),
                                        )
                                    )

                                    if verbose and line_no > 0:
                                        print(
                                            f"....Variable '{var_name}' defined at line {line_no}",
                                            file=sys.stderr,
                                        )
                                del mod.dataset  # type: ignore[attr-defined]
                            else:
                                if not silent:
                                    print(
                                        f"..No dataset found for {rel_path}",
                                        file=sys.stderr,
                                    )
                                continue
                        except SystemExit as e:
                            # Module attempted to exit (likely via argparse); mark as abort
                            if not silent:
                                print(
                                    f"..Module {rel_path} called SystemExit({e.code}) during import",
                                    file=sys.stderr,
                                )
                        except Exception as e:
                            # Re-raise other exceptions to be handled by outer except
                            print(f"..Error in {rel_path}: {e}", file=sys.stderr)
                            raise
                    finally:
                        sys.argv = prev_argv
            except (
                FileNotFoundError,
                ModuleNotFoundError,
            ) as e:  # pragma: no cover
                # Expected errors: missing output files or optional dependencies
                # These are common when executing dataset definitions outside their study environment
                if not silent:
                    error_type = (
                        "Missing file"
                        if isinstance(e, FileNotFoundError)
                        else "Missing module"
                    )
                    trace = get_trace(e)
                    print(
                        f"..{error_type} in {trace}: {e}",
                        file=sys.stderr,
                    )
            except KeyError as e:  # pragma: no cover
                # Catch KeyError from ehrql internals (e.g. missing codelists)
                if not silent:
                    print(
                        f"..KeyError in {rel_path}: {e}",
                        file=sys.stderr,
                    )
                    trace = get_trace(e)
                    if trace:
                        print(
                            f"\n  [KeyError] The dataset file might be trying to access a {e} property "
                            f"from a JSON file. The origin is:\n             {trace}\n"
                            "             Consider adding the property to spoofed_data.json.\n",
                            file=sys.stderr,
                        )
            except Exception as e:  # pragma: no cover
                # Catch other errors including ehrql FileValidationError (for missing data files)
                error_name = type(e).__name__
                # Common expected errors from ehrql
                if "Missing column" in str(e):
                    if not silent:
                        trace = get_trace(e)
                        print(
                            f"..Missing column in {trace}: {e}"
                            f"\n..YOU SHOULD ADD THESE COLUMNS TO spoofed_data/csv_data.csv",
                            file=sys.stderr,
                        )
                elif "FileValidationError" in error_name or "Missing file" in str(e):
                    if not silent:
                        trace = get_trace(e)
                        print(
                            f"..Missing data file in {trace}: {e}",
                            file=sys.stderr,
                        )
                else:
                    # Unexpected errors - show more detail
                    if not silent:
                        trace = get_trace(e)
                        print(
                            f"..Unexpected error in {trace}: {e}",
                            file=sys.stderr,
                        )

            if verbose:
                print(
                    f"....found {len(variables)} variables",
                    file=sys.stderr,
                )

    if not silent:
        print(
            f"..Collected {len(variables)} variables across {len(files)} dataset files",
            file=sys.stderr,
        )
    return variables


initial_modules: dict[str, types.ModuleType] = {}
initial_path: list[str] = []


def save_initial_module_snapshot() -> None:
    """Save the initial state of sys.modules and sys.path."""
    global initial_modules, initial_path
    initial_modules = sys.modules.copy()
    initial_path = sys.path.copy()


def reset_modules_to_snapshot() -> None:
    """Reset sys.modules to a previous snapshot, removing any new modules."""
    current_modules = set(sys.modules.keys())
    saved_modules = set(initial_modules.keys())

    sys.path = initial_path.copy()

    # Remove modules that were added since snapshot
    new_modules = current_modules - saved_modules
    for name in new_modules:
        try:
            del sys.modules[name]
        except KeyError:
            pass


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


def collect(
    output: str,
    repos: list[str] | None,
    silent: bool = False,
    verbose: bool = False,
) -> None:
    cache_dir = pathlib.Path(".ehrql_repo_cache")
    cache_dir.mkdir(exist_ok=True)

    items = code_search(verbose=verbose)

    if not silent:
        print(
            f"Found {len(items)} Python files mentioning create_dataset",
            file=sys.stderr,
        )

    grouped = group_items_by_repo(items)

    if not silent:
        print(f"Across {len(grouped)} repos", file=sys.stderr)

    global repo_name
    all_dataset_files: dict[str, (str, list[str], pathlib.Path)] = {}
    for repo_full, head_sha in grouped.items():
        owner, repo_name = repo_full.split("/", 1)

        if repos and repo_full not in repos and repo_name not in repos:
            if verbose:
                print(f"..Skipping {repo_full}", file=sys.stderr)
            continue

        if not silent:
            print(f"\n==> {repo_full} ({head_sha[:7]})", file=sys.stderr)

        repo_local_dir = cache_dir / f"{repo_name}-{head_sha[:8]}"
        if not repo_local_dir.exists():
            clone_url = f"https://github.com/{owner}/{repo_name}.git"
            cmd = [
                "git",
                "clone",
                "--depth",
                "1",
                clone_url,
                str(repo_local_dir),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0 and not silent:
                print(
                    f"..Clone failed {owner}/{repo_name}: {proc.stderr.strip()}",
                    file=sys.stderr,
                )
            else:
                if verbose:
                    print(
                        f"..Cloned {repo_full} to {repo_local_dir}",
                        file=sys.stderr,
                    )
        else:
            if verbose:
                print(f"..Using cached clone at {repo_local_dir}", file=sys.stderr)

        # NEW: Parse project.yaml to find dataset files
        dataset_files = parse_project_yaml(repo_local_dir)

        if dataset_files:
            all_dataset_files[repo_full] = (
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
            if not silent:
                print(
                    "..No ehrql generate_dataset commands found in project.yaml",
                    file=sys.stderr,
                )

    # Runtime enrichment
    if not silent:
        print("\nStarting runtime type enrichment phase", file=sys.stderr)

    cache_dir = pathlib.Path(".ehrql_repo_cache")
    cache_dir.mkdir(exist_ok=True)
    all_variables: list[VariableRecord] = []
    total_repos = len(all_dataset_files)
    current_repo_index = 0
    for repo, (head_sha, files, repo_local_dir) in all_dataset_files.items():
        current_repo_index += 1
        repo_name = repo
        if not silent:
            print(
                f"\nProcessing {repo} with {len(files)} dataset files... ({current_repo_index}/{total_repos} repos)",
                file=sys.stderr,
            )
        if not files:
            continue
        if verbose:
            print(f"..Enriching runtime types for {repo}", file=sys.stderr)

        all_variables.extend(
            get_runtime_dataset_variables(
                files,
                repo_local_dir,
                head_sha,
                silent=silent,
                verbose=verbose,
            )
        )

    # Write JSON with structure project -> dataset_file -> list of [variable_name, expression, permalink, series_type]
    out_map: dict[str, dict[str, Any]] = {}

    # qm_node analysis
    qm_out_map: dict[str, int] = {}

    # sort rows by project, file, variable name

    for r in sorted(
        all_variables, key=lambda r: (r.project_name, r.file_name, r.variable_name)
    ):
        # Remove any frozensets of codes or other things in case they only differ by this
        node_without_codes = re.sub(
            r"frozenset\(\{[^}]+\}\)", "<<FROZEN_SET>>", r.qm_node
        )
        expr_hash = hashlib.sha256(r.qm_node.encode("utf-8")).hexdigest()[:16]
        expr_hash_without_codes = hashlib.sha256(
            node_without_codes.encode("utf-8")
        ).hexdigest()[:16]
        qm_out_map[expr_hash_without_codes] = node_without_codes

        proj = r.project_name
        out_map.setdefault(proj, {})
        out_map[proj].setdefault("sha", r.project_sha)
        out_map[proj].setdefault("files", {})
        out_map[proj]["files"].setdefault(r.file_name, [])
        out_map[proj]["files"][r.file_name].append(
            [
                r.variable_name,
                r.series_type,
                r.line_no,
                expr_hash,
                expr_hash_without_codes,
            ]
        )

    json_data = {
        # current UK timestamp without milliseconds (i.e. BST or GMT rather than always UTC)
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "projects": out_map,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    with open("ehrql_qm_dump.json", "w", encoding="utf-8") as f:
        json.dump(qm_out_map, f, indent=2, ensure_ascii=False)

    if not silent:
        # print summary correctly counting the total number of variables given the structure of out_map
        total_vars = sum(
            len(vars_list)
            for proj_data in out_map.values()
            for vars_list in proj_data.get("files", {}).values()
        )
        print(
            f"\nWrote {output} with {total_vars} variables across {len(out_map)} projects and {sum(len(p.get('files', {})) for p in out_map.values())} dataset files",
            file=sys.stderr,
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect ehrql dataset variable definitions across the GitHub opensafely org"
    )
    p.add_argument(
        "--output", default="ehrql_variables.json", help="Output JSON file path"
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
    # args ends with an optional space separated list of repo names (e.g. "opensafely/pincer-measures opensafely/isaric-exploration")
    p.add_argument(
        "repos",
        nargs="*",
        help="list of repo names to process",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    setup_spoofs(silent=args.silent, verbose=args.verbose)

    save_initial_module_snapshot()

    try:
        collect(
            output=args.output,
            repos=args.repos,
            silent=args.silent,
            verbose=args.verbose,
        )
    except GitHubError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
