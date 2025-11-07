"""Test for ehrql_codelist_extractor.py"""

import pathlib
import tempfile

from parsing.ehrql_variable_extractor import extract_variable_codelists


def test_basic_codelist_extraction():
    """Test basic codelist_from_csv extraction from a simple dataset."""
    code = """
from ehrql import create_dataset, codelist_from_csv

# Define codelists
asthma_codes = codelist_from_csv("codelists/asthma.csv", column="code")
diabetes_codes = codelist_from_csv("codelists/diabetes.csv", column="code", system="snomed")

dataset = create_dataset()
dataset.has_asthma = patients.conditions.where(
    patients.conditions.code.is_in(asthma_codes)
).exists_for_patient()

dataset.has_diabetes = patients.conditions.where(
    patients.conditions.code.is_in(diabetes_codes)
).exists_for_patient()

dataset.age = patients.age_on("2024-01-01")
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = pathlib.Path(tmpdir)
        file_path = repo_root / "dataset_definition.py"
        file_path.write_text(code)

        codelists = extract_variable_codelists(file_path, repo_root)

        # has_asthma should reference asthma_codes
        assert "has_asthma" in codelists
        assert len(codelists["has_asthma"]) == 1
        assert codelists["has_asthma"][0][0] == "codelists/asthma.csv"
        assert "column=code" in codelists["has_asthma"][0]

        # has_diabetes should reference diabetes_codes
        assert "has_diabetes" in codelists
        assert len(codelists["has_diabetes"]) == 1
        assert codelists["has_diabetes"][0][0] == "codelists/diabetes.csv"
        assert "column=code" in codelists["has_diabetes"][0]
        assert "system=snomed" in codelists["has_diabetes"][0]

        # age should have no codelist calls
        assert "age" in codelists
        assert len(codelists["age"]) == 0


def test_no_codelists():
    """Test extraction when variables don't use codelists."""
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

        codelists = extract_variable_codelists(file_path, repo_root)

        # All variables should be present but with empty codelist arrays
        assert "age" in codelists
        assert len(codelists["age"]) == 0
        assert "sex" in codelists
        assert len(codelists["sex"]) == 0
        assert "region" in codelists
        assert len(codelists["region"]) == 0


def test_multiple_codelists_per_variable():
    """Test variable that references multiple codelists."""
    code = """
from ehrql import create_dataset, codelist_from_csv

copd_codes = codelist_from_csv("codelists/copd.csv", column="code")
emphysema_codes = codelist_from_csv("codelists/emphysema.csv", column="code")

dataset = create_dataset()
dataset.has_lung_disease = (
    patients.conditions.where(
        patients.conditions.code.is_in(copd_codes)
    ).exists_for_patient()
    | patients.conditions.where(
        patients.conditions.code.is_in(emphysema_codes)
    ).exists_for_patient()
)
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = pathlib.Path(tmpdir)
        file_path = repo_root / "dataset_definition.py"
        file_path.write_text(code)

        codelists = extract_variable_codelists(file_path, repo_root)

        # has_lung_disease should reference both codelists
        assert "has_lung_disease" in codelists
        assert len(codelists["has_lung_disease"]) == 2

        # Check both codelists are present (order may vary)
        codelist_files = [call[0] for call in codelists["has_lung_disease"]]
        assert "codelists/copd.csv" in codelist_files
        assert "codelists/emphysema.csv" in codelist_files


def test_codelist_via_intermediate_variable():
    """Test codelist extraction when variable uses an intermediate variable.

    This matches the pattern from opioids-covid-research where:
    - An intermediate variable references a codelist
    - A dataset variable uses case() with that intermediate variable
    """
    code = """
from ehrql import create_dataset, case, when, codelist_from_csv
from ehrql.tables.tpp import patients, clinical_events
import codelists

dataset = create_dataset()

# Intermediate variable that uses a codelist
ethnicity16 = clinical_events.where(
    clinical_events.snomedct_code.is_in(codelists.ethnicity_codes_16)
).where(
    clinical_events.date.is_on_or_before("2022-04-01")
).sort_by(
    clinical_events.date
).last_for_patient().snomedct_code.to_category(codelists.ethnicity_codes_16)

# Dataset variable that uses the intermediate variable
dataset.ethnicity16 = case(
    when(ethnicity16 == "1").then("White - British"),
    when(ethnicity16 == "2").then("White - Irish"),
    when(ethnicity16 == "3").then("White - Other"),
    otherwise="Unknown"
)
"""
    # Create codelists module
    codelists_code = """
from ehrql import codelist_from_csv

ethnicity_codes_16 = codelist_from_csv(
    "codelists/opensafely-ethnicity-snomed-0removed.csv",
    column="snomedcode",
    category_column="Grouping_16",
)
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = pathlib.Path(tmpdir)
        file_path = repo_root / "dataset_definition.py"
        file_path.write_text(code)

        # Create codelists.py module
        codelists_path = repo_root / "codelists.py"
        codelists_path.write_text(codelists_code)

        codelists = extract_variable_codelists(file_path, repo_root)

        # ethnicity16 should reference the codelist from the imported module
        assert "ethnicity16" in codelists
        assert len(codelists["ethnicity16"]) > 0, (
            "Should find at least one codelist call"
        )

        # Check that we found the ethnicity codelist
        codelist_files = [call[0] for call in codelists["ethnicity16"]]
        assert "codelists/opensafely-ethnicity-snomed-0removed.csv" in codelist_files


def test_codelist_deduplication():
    """Test that duplicate codelist references are deduplicated per variable.

    When a variable's expression tree references the same codelist multiple times
    (e.g., in multiple when() conditions or in intermediate variables), we should
    only report it once per variable.
    """
    code = """
from ehrql import create_dataset, case, when, codelist_from_csv
from ehrql.tables.tpp import patients, clinical_events
import codelists

dataset = create_dataset()

# Intermediate variable that uses the same codelist twice
ethnicity = clinical_events.where(
    clinical_events.snomedct_code.is_in(codelists.ethnicity_codes)
).where(
    clinical_events.date.is_on_or_before("2022-04-01")
).sort_by(
    clinical_events.date
).last_for_patient().snomedct_code.to_category(codelists.ethnicity_codes)

# Dataset variable with multiple when() conditions all referencing the same intermediate
# Each when() will trace back to ethnicity, which references the codelist twice
dataset.ethnicity_grouped = case(
    when(ethnicity == "1").then("Group A"),
    when(ethnicity == "2").then("Group A"),
    when(ethnicity == "3").then("Group A"),
    when(ethnicity == "4").then("Group B"),
    when(ethnicity == "5").then("Group B"),
    otherwise="Unknown"
)
"""
    # Create codelists module
    codelists_code = """
from ehrql import codelist_from_csv

ethnicity_codes = codelist_from_csv(
    "codelists/ethnicity.csv",
    column="code",
)
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = pathlib.Path(tmpdir)
        file_path = repo_root / "dataset_definition.py"
        file_path.write_text(code)

        # Create codelists.py module
        codelists_path = repo_root / "codelists.py"
        codelists_path.write_text(codelists_code)

        codelists = extract_variable_codelists(file_path, repo_root)

        # ethnicity_grouped references the codelist many times through the intermediate variable
        # but should only appear once in the output after deduplication
        assert "ethnicity_grouped" in codelists
        assert len(codelists["ethnicity_grouped"]) == 1, (
            f"Expected exactly 1 codelist after deduplication, got {len(codelists['ethnicity_grouped'])}"
        )

        # Verify it's the right codelist
        codelist_files = [call[0] for call in codelists["ethnicity_grouped"]]
        assert "codelists/ethnicity.csv" in codelist_files


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

    pytest.main([__file__, "-v"])
