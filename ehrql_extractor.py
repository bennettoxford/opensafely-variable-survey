"""Collect variable definitions from ehrql dataset definition files across a GitHub org.

See README.md for usage.
"""

from __future__ import annotations

import argparse
import builtins
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyarrow as _pa
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as _pa_ipc

from parsing.ehrql_github_helpers import (
    clone_repos,
    get_dataset_files,
    get_target_repos_and_shas,
)
from parsing.ehrql_qm_node_helpers import compact_qm_node
from parsing.ehrql_variable_extractor import extract_variable_line_numbers


def convert_spoofed_data(verbose: bool = False) -> int:
    root_dir = pathlib.Path(__file__).parent
    data_dir = root_dir / "spoofed_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    input_path = data_dir / "csv_data.csv"

    if verbose:
        print("Creating a .csv.gz and .arrow file from spoofed_data/csv_data.csv...")
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


@contextmanager
def working_directory(path):
    """Context manager for changing the working directory"""
    prev_cwd = pathlib.Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


@contextmanager
def suppress_output():
    """Suppress stdout and stderr output."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


@dataclass
class VariableRecord:
    project_name: str
    project_sha: str
    file_name: str
    variable_name: str
    series_type: str
    line_no: int | tuple[str, int]
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

            # NEVER redirect our internal cache files or output files
            if (
                file_str.endswith(".project_yaml_cache.json")
                or file_str.endswith("ehrql_variables.json")
                or file_str.endswith("ehrql_codelists.json")
            ):
                return builtin_open(file, mode, *args, **kwargs)

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

    # Also spoof Path.exists() for CSV files - ehrql's codelist_from_csv uses this
    original_exists = pathlib.Path.exists

    def spoofed_exists(self):
        path_str = str(self)
        if (
            path_str.endswith(".csv")
            or path_str.endswith(".csv.gz")
            or path_str.endswith(".arrow")
        ):
            if verbose:
                print(
                    f"..Spoofing exists() check for {path_str} -> True",
                    file=sys.stderr,
                )
            return True
        try:
            return original_exists(self)
        except (OSError, FileNotFoundError):
            return False

    pathlib.Path.exists = spoofed_exists

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


def get_runtime_dataset_variables(
    files: list[str],
    repo_root: pathlib.Path,
    head_sha: str,
    silent: bool = False,
    verbose: bool = False,
) -> set[str]:
    """
    Execute dataset definition files to extract runtime variable information.
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
        resolved_repo_root = repo_root.resolve()

        if verbose:
            print(f"..Collecting runtime variables for {abs_path}", file=sys.stderr)

        # Extract line numbers from AST before executing the module
        variable_line_numbers, variable_line_number_regexes = (
            extract_variable_line_numbers(abs_path, resolved_repo_root)
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
        # In fact let's keep on adding to the path until we get to the cache root
        parent_dir = abs_path.parent
        index = 0
        while parent_dir.name != CACHE_DIR:
            if str(parent_dir) in sys.path:
                sys.path.remove(str(parent_dir))
            sys.path.insert(index, str(parent_dir))
            index += 1
            parent_dir = parent_dir.parent

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
                            # Suppress output from print statements in dataset definitions unless verbose mode
                            if not verbose:
                                with suppress_output():
                                    spec.loader.exec_module(mod)  # type: ignore
                            else:
                                spec.loader.exec_module(mod)  # type: ignore
                            if hasattr(mod, "dataset"):
                                for var_name, series in mod.dataset._variables.items():  # type: ignore[attr-defined]
                                    # Get line number from AST parsing
                                    line_no: int | tuple[str, int] = (
                                        variable_line_numbers.get(var_name, -1)
                                    )

                                    # Check if we need to try regex matching
                                    # line_no is either -1 (not found), an int > 0, or a tuple
                                    if line_no == -1 and variable_line_number_regexes:
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
                                            qm_node=compact_qm_node(
                                                series._qm_node,
                                                max_depth=50
                                                if repo_name
                                                in [
                                                    "opensafely/polypharmacy-deprescribing-dementia",
                                                ]
                                                else None,
                                            ),
                                        )
                                    )

                                    if verbose:
                                        if isinstance(line_no, tuple):
                                            file_path, line_num = line_no
                                            print(
                                                f"....Variable '{var_name}' defined at {file_path}:{line_num}",
                                                file=sys.stderr,
                                            )
                                        elif line_no > 0:
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
                            "             Consider adding the property to spoofed_data/json_data.json.\n",
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


CACHE_DIR = ".ehrql_repo_cache"


def collect(
    output: str,
    repos: list[str] | None,
    silent: bool = False,
    verbose: bool = False,
    include_full_qm_node_dump: bool = False,
    force: bool = False,
) -> None:
    initial_start_time = time.time()
    cache_dir = pathlib.Path(CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)

    # Load existing results if not forcing recalculation
    existing_data: dict[str, dict[str, list]] = {}
    if not force and pathlib.Path(output).exists():
        try:
            with open(output, encoding="utf-8") as f:
                existing_json = json.load(f)
                existing_projects = existing_json.get("projects", {})

                # Build a map of (repo, sha) -> files_data
                # Include all processed SHAs, even those with no files_data
                for repo_full, repo_data in existing_projects.items():
                    sha = repo_data.get("sha")
                    files_data = repo_data.get("files", {})
                    if sha:
                        if repo_full not in existing_data:
                            existing_data[repo_full] = {}
                        existing_data[repo_full][sha] = files_data

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
        repos=repos,
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
        # cache_key = f"{repo}@{sha}"
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

    all_variables: list[VariableRecord] = []
    total_repos = len(all_dataset_files)
    current_repo_index = 0
    global repo_name
    start_time = time.time()
    for repo, (head_sha, files, repo_local_dir) in all_dataset_files.items():
        repo = repo.split("@")[0]  # strip off any @branch suffix
        current_repo_index += 1
        repo_name = repo
        if not silent:
            print(
                f"\nProcessing {repo} with {len(files)} dataset files... ({current_repo_index}/{total_repos} uncached repos)",
                file=sys.stderr,
            )
        if not files:
            continue

        # Start timing for this repo
        repo_start_time = time.time()
        variables_before = len(all_variables)

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

        # Print completion message with timing
        repo_duration = time.time() - repo_start_time
        variables_collected = len(all_variables) - variables_before
        if not silent:
            print(
                f"..Collected {variables_collected} variables across {len(files)} dataset files in {repo_duration:.1f}s",
                file=sys.stderr,
            )

    duration = time.time() - start_time
    if not silent:
        print(
            f"\nCompleted processing {len(all_dataset_files)} repos in {duration:.1f}s",
            file=sys.stderr,
        )
    write_start_time = time.time()
    # Write JSON with structure project -> dataset_file -> list of [variable_name, expression, permalink, series_type]
    out_map: dict[str, dict[str, Any]] = {}

    # First, add all cached results directly to out_map
    for repo_full, sha_dict in existing_data.items():
        for sha, files_data in sha_dict.items():
            out_map[repo_full] = {
                "sha": sha,
                "files": files_data,
            }

    if not silent and existing_data:
        cached_count = sum(len(shas) for shas in existing_data.values())
        print(
            f"\nAdded {cached_count} cached repo/SHA results to output",
            file=sys.stderr,
        )

    # qm_node analysis
    qm_out_map: dict[str, int] = {}

    # sort rows by project, file, variable name
    full_qm_out_map: dict[str, str] = {}

    for r in sorted(
        all_variables, key=lambda r: (r.project_name, r.file_name, r.variable_name)
    ):
        # expr_hash: full expression with sorted frozensets (stable ordering)
        # expr_hash_without_codes: frozensets replaced with placeholder for semantic comparison

        # First remove dates from the original node (keeping codes)
        node_without_dates = re.sub(r"datetime.date\([^)]+\)", "<<DATE>>", r.qm_node)
        # Then create a version without codes (for semantic comparison)
        node_without_codes_or_dates = re.sub(
            r"frozenset\(\{[^}]+\}\)", "<<FROZEN_SET>>", node_without_dates
        )

        expr_hash = hashlib.sha256(node_without_dates.encode("utf-8")).hexdigest()[:16]
        expr_hash_without_codes = hashlib.sha256(
            node_without_codes_or_dates.encode("utf-8")
        ).hexdigest()[:16]
        qm_out_map[expr_hash_without_codes] = node_without_codes_or_dates
        if include_full_qm_node_dump:
            # Also capture the full compacted node for debugging/diffing (use compacted version for determinism)
            full_qm_out_map[expr_hash] = r.qm_node

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
        json.dump(json_data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")  # ensure file ends with newline
    with open("ehrql_qm_dump.json", "w", encoding="utf-8") as f:
        json.dump(qm_out_map, f, indent=2, ensure_ascii=False, sort_keys=True)
    if include_full_qm_node_dump:
        # Write full (non-normalized) node dump keyed by full hash to aid debugging
        with open("ehrql_qm_full_dump.json", "w", encoding="utf-8") as f:
            json.dump(full_qm_out_map, f, indent=2, ensure_ascii=False, sort_keys=True)

    if not silent:
        write_duration = time.time() - write_start_time
        print(
            f"\nHashed nodes and wrote output files in {write_duration:.1f}s",
            file=sys.stderr,
        )
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
        total_duration = time.time() - initial_start_time
        print(
            f"\nTotal execution time: {total_duration:.1f}s",
            file=sys.stderr,
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect ehrql dataset variable definitions across the GitHub opensafely org"
    )
    p.add_argument(
        "--output", default="data/ehrql_variables.json", help="Output JSON file path"
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
        "--include-full-qm-node-dump",
        action="store_true",
        help="Include full (non-normalized) node dump in output. This is many GB and only useful for debugging.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force recalculation of all results, ignoring cached data from previous runs",
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

    collect(
        output=args.output,
        repos=args.repos,
        silent=args.silent,
        verbose=args.verbose,
        include_full_qm_node_dump=args.include_full_qm_node_dump,
        force=args.force,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
