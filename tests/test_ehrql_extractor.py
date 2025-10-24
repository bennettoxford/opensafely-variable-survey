"""Comprehensive pytest test suite for ehrql_extractor.py.

Tests all four passes of variable detection:
- Pass 1: Dataset assignment from function calls
- Pass 2: Direct module-level dataset operations
- Pass 3: Loop-based dynamic variables
- Pass 4: Non-loop helper function calls
"""

import pathlib
import tempfile

import pytest

from parsing.ehrql_variable_extractor import extract_variable_line_numbers


class TestPass1LocalFunction:
    """Test Pass 1: Dataset creation tracing with local functions."""

    def test_direct_assignment(self):
        """Test dataset.var = expr inside a local function."""
        code = """
from ehrql import create_dataset

def create_my_dataset():
    dataset = create_dataset()
    dataset.age = patients.age_on("2024-01-01")
    dataset.sex = patients.sex
    return dataset

dataset = create_my_dataset()
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert "age" in line_numbers
            assert "sex" in line_numbers
            assert line_numbers["age"] == 6
            assert line_numbers["sex"] == 7
            assert regexes == []

    def test_add_column(self):
        """Test dataset.add_column() inside a local function."""
        code = """
from ehrql import create_dataset

def create_my_dataset():
    dataset = create_dataset()
    dataset.add_column("age", patients.age_on("2024-01-01"))
    dataset.add_column("sex", patients.sex)
    return dataset

dataset = create_my_dataset()
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert "age" in line_numbers
            assert "sex" in line_numbers
            assert line_numbers["age"] == 6
            assert line_numbers["sex"] == 7
            assert regexes == []

    def test_mixed_patterns(self):
        """Test mix of dataset.var and dataset.add_column() in local function."""
        code = """
from ehrql import create_dataset

def build_dataset():
    dataset = create_dataset()
    dataset.age = patients.age_on("2024-01-01")
    dataset.add_column("sex", patients.sex)
    dataset.region = patients.region
    return dataset

dataset = build_dataset()
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert "age" in line_numbers
            assert "sex" in line_numbers
            assert "region" in line_numbers
            assert line_numbers["age"] == 6
            assert line_numbers["sex"] == 7
            assert line_numbers["region"] == 8


class TestPass1CrossFile:
    """Test Pass 1: Dataset creation tracing with imported functions (cross-file)."""

    def test_imported_direct_assignment(self):
        """Test dataset.var = expr in imported function (cross-file)."""
        helper_code = """
from ehrql import create_dataset

def build_dataset():
    dataset = create_dataset()
    dataset.age = patients.age_on("2024-01-01")
    dataset.sex = patients.sex
    return dataset
"""
        main_code = """
from helpers import build_dataset

dataset = build_dataset()
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            helper_path = repo_root / "helpers.py"
            helper_path.write_text(helper_code)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(main_code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert "age" in line_numbers
            assert "sex" in line_numbers
            assert isinstance(line_numbers["age"], tuple)
            assert isinstance(line_numbers["sex"], tuple)
            assert line_numbers["age"][0] == "helpers.py"
            assert line_numbers["sex"][0] == "helpers.py"
            assert line_numbers["age"][1] == 6
            assert line_numbers["sex"][1] == 7

    def test_imported_add_column(self):
        """Test dataset.add_column() in imported function (cross-file)."""
        helper_code = """
from ehrql import create_dataset

def build_dataset():
    dataset = create_dataset()
    dataset.add_column("age", patients.age_on("2024-01-01"))
    dataset.add_column("sex", patients.sex)
    return dataset
"""
        main_code = """
from helpers import build_dataset

dataset = build_dataset()
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            helper_path = repo_root / "helpers.py"
            helper_path.write_text(helper_code)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(main_code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert "age" in line_numbers
            assert "sex" in line_numbers
            assert isinstance(line_numbers["age"], tuple)
            assert isinstance(line_numbers["sex"], tuple)
            assert line_numbers["age"][0] == "helpers.py"
            assert line_numbers["sex"][0] == "helpers.py"
            assert line_numbers["age"][1] == 6
            assert line_numbers["sex"][1] == 7

    def test_nested_subdirectory_import(self):
        """Test import from parent directory helper."""
        helper_code = """
from ehrql import create_dataset

def build_dataset():
    dataset = create_dataset()
    dataset.age = patients.age_on("2024-01-01")
    dataset.sex = patients.sex
    return dataset
"""
        main_code = """
from analysis.utils import build_dataset

dataset = build_dataset()
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            analysis_dir = repo_root / "analysis"
            analysis_dir.mkdir()
            helper_path = analysis_dir / "utils.py"
            helper_path.write_text(helper_code)

            file_path = analysis_dir / "dataset_definition.py"
            file_path.write_text(main_code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert "age" in line_numbers
            assert "sex" in line_numbers
            assert isinstance(line_numbers["age"], tuple)
            assert isinstance(line_numbers["sex"], tuple)
            assert line_numbers["age"][0] == "analysis/utils.py"
            assert line_numbers["sex"][0] == "analysis/utils.py"

    def test_dict_return_and_setattr_loop(self):
        """Test generate_variables returns dict and generate_dataset setattr loop assigns to dataset.

        Structure:
        - dataset_definition_prevax.py calls generate_dataset from dataset_definition_cohorts.py
        - generate_dataset imports generate_variables from variables_cohorts.py
        - generate_variables returns a dict {"alpha": expr, "beta": expr}
        - generate_dataset iterates variables.items() and does setattr(dataset, var_name, var_value)
        Expectation: extractor returns ('variables_cohorts.py', dict_line) for alpha and beta.
        """
        variables_code = """
def generate_variables(a, b, c):
    return {
        "alpha": patients.sex,
        "beta": patients.age,
    }
"""
        cohorts_code = """
from ehrql import create_dataset

def generate_dataset(index_date, end_date_exp, end_date_out):
    dataset = create_dataset()
    from variables_cohorts import generate_variables
    variables = generate_variables(index_date, end_date_exp, end_date_out)
    for var_name, var_value in variables.items():
        setattr(dataset, var_name, var_value)
    return dataset
"""
        prevax_code = """
from dataset_definition_cohorts import generate_dataset

dataset = generate_dataset(1, 2, 3)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            (repo_root / "variables_cohorts.py").write_text(variables_code)
            (repo_root / "dataset_definition_cohorts.py").write_text(cohorts_code)
            file_path = repo_root / "dataset_definition_prevax.py"
            file_path.write_text(prevax_code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert "alpha" in line_numbers
            assert "beta" in line_numbers
            # Cross-file: should be tuples pointing to variables file where dict is defined
            assert isinstance(line_numbers["alpha"], tuple)
            assert isinstance(line_numbers["beta"], tuple)
            assert line_numbers["alpha"][0] == "variables_cohorts.py"
            assert line_numbers["beta"][0] == "variables_cohorts.py"
            # Dict literal is on line 3 in variables_code above (leading newline in triple-quoted string)
            assert line_numbers["alpha"][1] == 3
            assert line_numbers["beta"][1] == 3

    def test_dict_constructor_and_setattr_loop_cov_bin_ckd(self):
        """Test generate_variables builds dict(...) and generate_dataset setattr loop assigns to dataset.

        Structure mirrors attached repo:
        - dataset_definition_prevax.py calls generate_dataset from dataset_definition_cohorts.py
        - generate_dataset imports generate_variables from variables_cohorts.py
        - generate_variables constructs dynamic_variables via dict(cov_bin_ckd=..., ...)
        - generate_dataset iterates variables.items() and does setattr(dataset, var_name, var_value)
        Expectation: extractor returns ('variables_cohorts.py', definition_line) for cov_bin_ckd.
        """
        variables_code = """
def generate_variables(index_date, end_date_exp, end_date_out):
    cov_bin_ckd = patients.something
    cov_bin_diabetes = patients.otherthing
    dynamic_variables = dict(
        cov_bin_ckd=cov_bin_ckd,
        cov_bin_diabetes=cov_bin_diabetes,
    )
    return dynamic_variables
"""
        cohorts_code = """
from ehrql import create_dataset

def generate_dataset(index_date, end_date_exp, end_date_out):
    dataset = create_dataset()
    from variables_cohorts import generate_variables
    variables = generate_variables(index_date, end_date_exp, end_date_out)
    for var_name, var_value in variables.items():
        setattr(dataset, var_name, var_value)
    return dataset
"""
        prevax_code = """
from dataset_definition_cohorts import generate_dataset

dataset = generate_dataset(1, 2, 3)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            (repo_root / "variables_cohorts.py").write_text(variables_code)
            (repo_root / "dataset_definition_cohorts.py").write_text(cohorts_code)
            file_path = repo_root / "dataset_definition_prevax.py"
            file_path.write_text(prevax_code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert "cov_bin_ckd" in line_numbers
            assert isinstance(line_numbers["cov_bin_ckd"], tuple)
            assert line_numbers["cov_bin_ckd"][0] == "variables_cohorts.py"
            # cov_bin_ckd is defined on line 3 in variables_code above (leading newline in triple-quoted string)
            assert line_numbers["cov_bin_ckd"][1] == 3
            assert "cov_bin_diabetes" in line_numbers
            assert line_numbers["cov_bin_diabetes"][1] == 4


class TestPass2ModuleLevel:
    """Test Pass 2: Direct module-level dataset operations."""

    def test_direct_attribute_assignment(self):
        """Test dataset.variable_name = expression at module level."""
        code = """
from ehrql import create_dataset

dataset = create_dataset()
dataset.age = patients.age_on("2024-01-01")
dataset.sex = patients.sex
dataset.region = patients.region
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert "age" in line_numbers
            assert "sex" in line_numbers
            assert "region" in line_numbers
            assert line_numbers["age"] == 5
            assert line_numbers["sex"] == 6
            assert line_numbers["region"] == 7
            assert regexes == []

    def test_add_column_string_literal(self):
        """Test dataset.add_column("name", expr) at module level."""
        code = """
from ehrql import create_dataset

dataset = create_dataset()
dataset.add_column("age", patients.age_on("2024-01-01"))
dataset.add_column("sex", patients.sex)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert "age" in line_numbers
            assert "sex" in line_numbers
            assert line_numbers["age"] == 5
            assert line_numbers["sex"] == 6
            assert regexes == []

    def test_add_column_fstring(self):
        """Test dataset.add_column(f"name_{var}", expr) at module level."""
        code = """
from ehrql import create_dataset

dataset = create_dataset()
suffix = "_baseline"
dataset.add_column(f"age{suffix}", patients.age_on("2024-01-01"))
dataset.add_column(f"sex{suffix}", patients.sex)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) == 2
            patterns = [r[0] for r in regexes]
            assert any("age" in p for p in patterns)
            assert any("sex" in p for p in patterns)
            lines = [r[1] for r in regexes]
            assert 6 in lines
            assert 7 in lines

    def test_mixed_patterns(self):
        """Test mix of direct assignment, add_column, and f-strings at module level."""
        code = """
from ehrql import create_dataset

dataset = create_dataset()
dataset.age = patients.age_on("2024-01-01")
dataset.add_column("sex", patients.sex)
suffix = "_test"
dataset.add_column(f"region{suffix}", patients.region)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert "age" in line_numbers
            assert "sex" in line_numbers
            assert line_numbers["age"] == 5
            assert line_numbers["sex"] == 6
            assert len(regexes) == 1
            assert "region" in regexes[0][0]
            assert regexes[0][1] == 8


class TestPass3LoopDetection:
    """Test Pass 3: Loop-based dynamic variables."""

    def test_loop_fstring(self):
        """Test dataset.add_column(f"name_{i}", expr) in loop."""
        code = """
from ehrql import create_dataset

dataset = create_dataset()

for i in range(5):
    dataset.add_column(f"region_{i}", "some_expression")
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 1
            patterns = [r[0] for r in regexes]
            assert any("region" in p for p in patterns)
            assert regexes[0][1] == 7  # Loop line

    def test_loop_setattr(self):
        """Test setattr(dataset, f"var_{i}", expr) in loop."""
        code = """
from ehrql import create_dataset

dataset = create_dataset()

for i in range(3):
    setattr(dataset, f"var_{i}", "some_expression")
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 1
            patterns = [r[0] for r in regexes]
            assert any("var" in p for p in patterns)
            assert regexes[0][1] == 6

    def test_loop_subscript(self):
        """Test dataset[f"name_{i}"] = expr in loop."""
        code = """
from ehrql import create_dataset

dataset = create_dataset()

for name in ["alpha", "beta", "gamma"]:
    dataset[f"test_{name}"] = "some_expression"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 1
            patterns = [r[0] for r in regexes]
            assert any("test" in p for p in patterns)
            assert regexes[0][1] == 6

    def test_loop_local_helper_positional(self):
        """Test local helper called with positional args in loop."""
        code = """
from ehrql import create_dataset

def add_vars(ds, suffix):
    ds.add_column(f"age{suffix}", patients.age)
    ds.add_column(f"sex{suffix}", patients.sex)

dataset = create_dataset()

for i in range(3):
    add_vars(dataset, f"_{i}")
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 2
            patterns = [r[0] for r in regexes]
            assert any("age" in p for p in patterns)
            assert any("sex" in p for p in patterns)
            # Now returns lines from inside the helper (5, 6), not the loop line (10)
            lines = [r[1] for r in regexes]
            assert 5 in lines
            assert 6 in lines

    def test_loop_imported_helper(self):
        """Test imported helper called in loop."""
        helper_code = """
def add_region(ds, idx):
    ds.add_column(f"region_{idx}", expr)
"""
        main_code = """
from ehrql import create_dataset
from helpers import add_region

dataset = create_dataset()

for i in range(5):
    add_region(dataset, i)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            helper_path = repo_root / "helpers.py"
            helper_path.write_text(helper_code)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(main_code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 1
            patterns = [r[0] for r in regexes]
            assert any("region" in p for p in patterns)

    def test_loop_star_import_helper(self):
        """Test star-imported helper called in loop with keyword args."""
        helper_code = """
def demographic_variables(dataset, index_date, var_name_suffix=""):
    dataset.add_column(f"age{var_name_suffix}", patients.age_on(index_date))
    dataset.add_column(f"sex{var_name_suffix}", patients.sex)
"""
        main_code = """
from ehrql import create_dataset
from helpers import *

dataset = create_dataset()

for i in range(3):
    suffix = f"_{i}"
    demographic_variables(dataset=dataset, index_date="2024-01-01", var_name_suffix=suffix)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            helper_path = repo_root / "helpers.py"
            helper_path.write_text(helper_code)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(main_code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 2
            patterns = [r[0] for r in regexes]
            assert any("age" in p for p in patterns)
            assert any("sex" in p for p in patterns)


class TestPass4StandaloneHelpers:
    """Test Pass 4: Non-loop helper function calls at module level."""

    def test_local_helper_positional(self):
        """Test local helper with positional args at module level."""
        code = """
from ehrql import create_dataset

def demographic_variables(ds, date, suffix=""):
    ds.add_column(f"age{suffix}", patients.age_on(date))
    ds.add_column(f"sex{suffix}", patients.sex)

dataset = create_dataset()
demographic_variables(dataset, "2024-01-01")
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 2
            patterns = [r[0] for r in regexes]
            assert any("age" in p for p in patterns)
            assert any("sex" in p for p in patterns)

    def test_local_helper_keyword(self):
        """Test local helper with keyword args at module level."""
        code = """
from ehrql import create_dataset

def demographic_variables(ds, date, suffix=""):
    ds.add_column(f"age{suffix}", patients.age_on(date))
    ds.add_column(f"sex{suffix}", patients.sex)

dataset = create_dataset()
demographic_variables(ds=dataset, date="2024-01-01")
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 2
            patterns = [r[0] for r in regexes]
            assert any("age" in p for p in patterns)
            assert any("sex" in p for p in patterns)

    def test_imported_helper_keyword(self):
        """Test imported helper with keyword args."""
        helper_code = """
def demographic_variables(dataset, index_date, var_name_suffix=""):
    dataset.add_column(f"age{var_name_suffix}", patients.age_on(index_date))
    dataset.add_column(f"sex{var_name_suffix}", patients.sex)
"""
        main_code = """
from ehrql import create_dataset
from helpers import demographic_variables

dataset = create_dataset()
demographic_variables(dataset=dataset, index_date="2024-01-01")
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            helper_path = repo_root / "helpers.py"
            helper_path.write_text(helper_code)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(main_code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 2
            patterns = [r[0] for r in regexes]
            assert any("age" in p for p in patterns)
            assert any("sex" in p for p in patterns)

    def test_star_import_helper(self):
        """Test star-imported helper at module level."""
        helper_code = """
def add_vars(dataset, suffix=""):
    dataset.add_column(f"var_a{suffix}", expr)
    dataset.add_column(f"var_b{suffix}", expr)
"""
        main_code = """
from ehrql import create_dataset
from helpers import *

dataset = create_dataset()
add_vars(dataset)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            helper_path = repo_root / "helpers.py"
            helper_path.write_text(helper_code)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(main_code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 2
            patterns = [r[0] for r in regexes]
            assert any("var_a" in p for p in patterns)
            assert any("var_b" in p for p in patterns)

    def test_module_function_call(self):
        """Test import module; module.func(dataset, ...)."""
        helper_code = """
def add_demographic_vars(dataset, suffix=""):
    dataset.add_column(f"age{suffix}", patients.age)
    dataset.add_column(f"sex{suffix}", patients.sex)
    dataset.add_column(f"region{suffix}", patients.region)
"""
        main_code = """
from ehrql import create_dataset
import helpers

dataset = create_dataset()
helpers.add_demographic_vars(dataset, "_baseline")
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            helper_path = repo_root / "helpers.py"
            helper_path.write_text(helper_code)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(main_code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 3
            patterns = [r[0] for r in regexes]
            assert any("age" in p for p in patterns)
            assert any("sex" in p for p in patterns)
            assert any("region" in p for p in patterns)

    def test_module_function_in_loop(self):
        """Test import module; module.func() in loop."""
        helper_code = """
def add_region_vars(dataset, region_id):
    dataset.add_column(f"age_region_{region_id}", patients.age)
    dataset.add_column(f"sex_region_{region_id}", patients.sex)
"""
        main_code = """
from ehrql import create_dataset
import helpers

dataset = create_dataset()

for region in range(5):
    helpers.add_region_vars(dataset, region)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            helper_path = repo_root / "helpers.py"
            helper_path.write_text(helper_code)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(main_code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 2
            patterns = [r[0] for r in regexes]
            assert any("age_region_" in p for p in patterns)
            assert any("sex_region_" in p for p in patterns)

    def test_module_alias(self):
        """Test import module as alias; alias.func()."""
        helper_code = """
def add_columns(dataset, prefix):
    dataset.add_column(f"{prefix}_age", patients.age)
    dataset.add_column(f"{prefix}_sex", patients.sex)
"""
        main_code = """
from ehrql import create_dataset
import helpers as h

dataset = create_dataset()

for prefix in ["cohort1", "cohort2"]:
    h.add_columns(dataset, prefix)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            helper_path = repo_root / "helpers.py"
            helper_path.write_text(helper_code)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(main_code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 2
            patterns = [r[0] for r in regexes]
            assert any("_age" in p for p in patterns)
            assert any("_sex" in p for p in patterns)


class TestPositionalArguments:
    """Test helper functions with positional arguments in various contexts."""

    def test_positional_imported_helper_loop(self):
        """Test imported helper with positional args in loop."""
        helper_code = """
def create_cohort_vars(dataset, cohort_name):
    dataset.add_column(f"age_{cohort_name}", patients.age)
    dataset.add_column(f"sex_{cohort_name}", patients.sex)
"""
        main_code = """
from ehrql import create_dataset
from helpers import create_cohort_vars

dataset = create_dataset()

for cohort in ["baseline", "followup"]:
    create_cohort_vars(dataset, cohort)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            helper_path = repo_root / "helpers.py"
            helper_path.write_text(helper_code)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(main_code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 2
            patterns = [r[0] for r in regexes]
            assert any("age_" in p for p in patterns)
            assert any("sex_" in p for p in patterns)

    def test_mixed_positional_keyword_args(self):
        """Test helper with mix of positional and keyword arguments."""
        code = """
from ehrql import create_dataset

def add_variables(ds, date, prefix="", suffix=""):
    ds.add_column(f"{prefix}age{suffix}", patients.age_on(date))
    ds.add_column(f"{prefix}sex{suffix}", patients.sex)

dataset = create_dataset()

for i in range(3):
    add_variables(dataset, "2024-01-01", suffix=f"_{i}")
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 2
            patterns = [r[0] for r in regexes]
            assert any("age" in p for p in patterns)
            assert any("sex" in p for p in patterns)

    def test_positional_star_import(self):
        """Test star-imported helper with positional args."""
        helper_code = """
def demographic_variables(dataset, index_date, var_name_suffix=""):
    dataset.add_column(f"age{var_name_suffix}", patients.age_on(index_date))
    dataset.add_column(f"sex{var_name_suffix}", patients.sex)
"""
        main_code = """
from ehrql import create_dataset
from helpers import *

dataset = create_dataset()

for i in range(2):
    demographic_variables(dataset, "2024-01-01", f"_{i}")
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            helper_path = repo_root / "helpers.py"
            helper_path.write_text(helper_code)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(main_code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 2
            patterns = [r[0] for r in regexes]
            assert any("age" in p for p in patterns)
            assert any("sex" in p for p in patterns)


@pytest.mark.skip(reason="BinOp concatenation not yet implemented")
class TestBinOpConcatenation:
    """Test BinOp string concatenation (NOT CURRENTLY SUPPORTED).

    These tests document expected behavior for future implementation.
    BinOp concatenation ("prefix" + variable) does not work even though
    code exists in ehrql_extractor.py.
    """

    def test_binop_in_loop(self):
        """Test dataset.add_column("prefix" + var, expr) in loop."""
        code = """
from ehrql import create_dataset

dataset = create_dataset()

for suffix in ["_pre", "_post", "_during"]:
    dataset.add_column("outcome" + suffix, patients.some_value)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 1
            patterns = [r[0] for r in regexes]
            assert any("outcome" in p for p in patterns)

    def test_binop_in_helper(self):
        """Test BinOp in helper function."""
        code = """
from ehrql import create_dataset

def add_vars(ds, prefix, suffix):
    ds.add_column("age_" + prefix + suffix, patients.age)
    ds.add_column("sex_" + prefix + suffix, patients.sex)

dataset = create_dataset()

for i in range(3):
    add_vars(dataset, "cohort", f"_{i}")
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            file_path = repo_root / "dataset_definition.py"
            file_path.write_text(code)

            line_numbers, regexes = extract_variable_line_numbers(file_path, repo_root)

            assert len(regexes) >= 2
            patterns = [r[0] for r in regexes]
            assert any("age_" in p for p in patterns)
            assert any("sex_" in p for p in patterns)


class TestImportedSetattr:
    """Test helper functions that use setattr on a dataset parameter."""

    def test_imported_setattr_function(self):
        """Test function that takes dataset as param and uses setattr (like has_prior_comorbidity)."""
        # variables.py helper file
        variables_code = """
from ehrql import days

def has_prior_comorbidity(extract_name, codelist_name, system, column_name, dataset):
    if system == "snomed":
        characteristic = (
            clinical_events.where(clinical_events.snomedct_code.is_in(codelist))
            .where(clinical_events.date.is_on_or_before(getattr(dataset, column_name) - days(1)))
            .exists_for_patient()
        )
        setattr(dataset, extract_name, characteristic)

    if system == "ctv3":
        characteristic = (
            clinical_events.where(clinical_events.ctv3_code.is_in(codelist))
            .where(clinical_events.date.is_on_or_before(getattr(dataset, column_name) - days(1)))
            .exists_for_patient()
        )
        setattr(dataset, extract_name, characteristic)
"""
        # Main dataset definition file
        main_code = """
from ehrql import Dataset
from variables import has_prior_comorbidity

dataset = Dataset()
dataset.first_admission_date = patients.date_of_birth

# These calls should be detected - the line number should be the call site
has_prior_comorbidity("ccd_pc", "chronic_cardiac_disease", "snomed", "first_admission_date", dataset)
has_prior_comorbidity("hypertension_pc", "hypertension", "snomed", "first_admission_date", dataset)
has_prior_comorbidity("copd_pc", "copd", "snomed", "first_admission_date", dataset)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)

            # Create variables.py file
            variables_file = repo_root / "variables.py"
            variables_file.write_text(variables_code)

            # Create main dataset definition file
            main_file = repo_root / "dataset_definition.py"
            main_file.write_text(main_code)

            line_numbers, regexes = extract_variable_line_numbers(main_file, repo_root)

            # Should detect all three variables
            assert "ccd_pc" in line_numbers
            assert "hypertension_pc" in line_numbers
            assert "copd_pc" in line_numbers

            # Line numbers should be from the call site in main file, not setattr in variables.py
            assert (
                line_numbers["ccd_pc"] == 9
            )  # Line of first has_prior_comorbidity call
            assert line_numbers["hypertension_pc"] == 10  # Line of second call
            assert line_numbers["copd_pc"] == 11  # Line of third call

            assert regexes == []
