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


def test_codelist_from_enum_with_fstring():
    """Test codelist extraction when codelist is defined in an Enum with f-string path.

    This matches the pattern from pincer-measures where:
    - Codelists are defined as Enum members
    - Each enum member constructs the CSV path using an f-string
    - Variables access codelists via Enum.MEMBER.codes
    """
    code = """
from ehrql import Dataset, months
from ehrql.tables.beta.core import medications
from codelists import Codelists
from utils import HistoricalEvent

INTERVAL = months(3).starting_on("2024-01-01")
dataset = Dataset()

hist_med = HistoricalEvent("medication", interval=INTERVAL)

oral_nsaid = hist_med.fetch(Codelists.ORAL_NSAID.codes, 3)
dataset.oral_nsaid = oral_nsaid.exists_for_patient()

ppi = hist_med.fetch(Codelists.ULCER_HEALING_DRUGS.codes, 3)
dataset.ppi = ppi.exists_for_patient()
"""
    # Create codelists module with Enum pattern
    codelists_code = """
from enum import Enum
from ehrql.codes import codelist_from_csv


class Codelists(Enum):
    ULCER_HEALING_DRUGS = ("pincer-ppi", "id")
    ORAL_NSAID = ("pincer-nsaid", "id")
    PEPTIC_ULCER = ("pincer-pep", "code")

    def __init__(self, codelist_name: str, column: str) -> None:
        self.codelist_name = codelist_name
        self._column = column
        self.codes = codelist_from_csv(
            f"codelists/{self.codelist_name}.csv", column=self._column
        )
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = pathlib.Path(tmpdir)
        file_path = repo_root / "dataset_definition.py"
        file_path.write_text(code)

        # Create codelists.py module
        codelists_path = repo_root / "codelists.py"
        codelists_path.write_text(codelists_code)

        codelists_result = extract_variable_codelists(file_path, repo_root)

        # oral_nsaid should find the codelist from the Enum
        assert "oral_nsaid" in codelists_result
        assert len(codelists_result["oral_nsaid"]) == 1, (
            f"Expected 1 codelist for oral_nsaid, got {len(codelists_result['oral_nsaid'])}"
        )

        # Check the CSV path is correctly extracted
        codelist_files = [call[0] for call in codelists_result["oral_nsaid"]]
        assert "codelists/pincer-nsaid.csv" in codelist_files, (
            f"Expected 'codelists/pincer-nsaid.csv' in {codelist_files}"
        )

        # Check the column parameter
        oral_nsaid_params = codelists_result["oral_nsaid"][0]
        assert "column=id" in oral_nsaid_params, (
            f"Expected 'column=id' in {oral_nsaid_params}"
        )

        # ppi should also find its codelist
        assert "ppi" in codelists_result
        assert len(codelists_result["ppi"]) == 1
        codelist_files = [call[0] for call in codelists_result["ppi"]]
        assert "codelists/pincer-ppi.csv" in codelist_files


def test_codelist_via_dataset_variable_reference():
    """Test codelist extraction when a variable references another dataset variable.

    This matches the pattern from pincer-measures where:
    - dataset.ppi uses a codelist
    - dataset.indicator_a_denominator references dataset.ppi
    - We should trace through dataset.ppi to find its codelists
    """
    code = """
from ehrql import Dataset, months
from ehrql.tables.beta.core import medications
from codelists import Codelists
from utils import HistoricalEvent

INTERVAL = months(3).starting_on("2024-01-01")
dataset = Dataset()

hist_med = HistoricalEvent("medication", interval=INTERVAL)

# Base variable that uses a codelist
ppi = hist_med.fetch(Codelists.ULCER_HEALING_DRUGS.codes, 3)
dataset.ppi = ppi.exists_for_patient()

oral_nsaid = hist_med.fetch(Codelists.ORAL_NSAID.codes, 3)
dataset.oral_nsaid = oral_nsaid.exists_for_patient()

# Composite variable that references other dataset variables
dataset.indicator_a_denominator = dataset.population_filter & ~dataset.ppi

dataset.indicator_a_numerator = dataset.indicator_a_denominator & dataset.oral_nsaid
"""
    # Create codelists module with Enum pattern
    codelists_code = """
from enum import Enum
from ehrql.codes import codelist_from_csv


class Codelists(Enum):
    ULCER_HEALING_DRUGS = ("pincer-ppi", "id")
    ORAL_NSAID = ("pincer-nsaid", "id")

    def __init__(self, codelist_name: str, column: str) -> None:
        self.codelist_name = codelist_name
        self._column = column
        self.codes = codelist_from_csv(
            f"codelists/{self.codelist_name}.csv", column=self._column
        )
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = pathlib.Path(tmpdir)
        file_path = repo_root / "dataset_definition.py"
        file_path.write_text(code)

        # Create codelists.py module
        codelists_path = repo_root / "codelists.py"
        codelists_path.write_text(codelists_code)

        codelists_result = extract_variable_codelists(file_path, repo_root)

        # indicator_a_denominator should find the codelist from dataset.ppi
        assert "indicator_a_denominator" in codelists_result
        assert len(codelists_result["indicator_a_denominator"]) == 1, (
            f"Expected 1 codelist for indicator_a_denominator (from dataset.ppi), "
            f"got {len(codelists_result['indicator_a_denominator'])}: {codelists_result['indicator_a_denominator']}"
        )

        # Check it found the ppi codelist
        codelist_files = [
            call[0] for call in codelists_result["indicator_a_denominator"]
        ]
        assert "codelists/pincer-ppi.csv" in codelist_files, (
            f"Expected 'codelists/pincer-ppi.csv' in {codelist_files}"
        )

        # indicator_a_numerator should find codelists from both dataset.ppi and dataset.oral_nsaid
        assert "indicator_a_numerator" in codelists_result
        assert len(codelists_result["indicator_a_numerator"]) == 2, (
            f"Expected 2 codelists for indicator_a_numerator (ppi + oral_nsaid), "
            f"got {len(codelists_result['indicator_a_numerator'])}: {codelists_result['indicator_a_numerator']}"
        )

        # Check it found both codelists
        codelist_files = [call[0] for call in codelists_result["indicator_a_numerator"]]
        assert "codelists/pincer-ppi.csv" in codelist_files, (
            f"Expected 'codelists/pincer-ppi.csv' in {codelist_files}"
        )
        assert "codelists/pincer-nsaid.csv" in codelist_files, (
            f"Expected 'codelists/pincer-nsaid.csv' in {codelist_files}"
        )


def test_inline_literal_codelist_detection():
    """Inline Python lists of codes should be surfaced as codelists."""

    dataset_code = """
from ehrql import create_dataset
import codelists

dataset = create_dataset()


def next_emergency_attendance(index_date, diagnoses_contains_any_of=None):
    return diagnoses_contains_any_of


dataset.covid_emergency_date = next_emergency_attendance(
    "2023-01-01", codelists.covid_emergency_codes
)
"""

    codelists_code = """
covid_emergency_codes = [
    "1240751000000100",
    "1325171000000109",
    "1325181000000106",
]
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = pathlib.Path(tmpdir)
        file_path = repo_root / "dataset_definition.py"
        file_path.write_text(dataset_code)

        codelists_path = repo_root / "codelists.py"
        codelists_path.write_text(codelists_code)

        codelists = extract_variable_codelists(file_path, repo_root)

        assert "covid_emergency_date" in codelists
        calls = codelists["covid_emergency_date"]
        assert len(calls) == 1, "Should detect inline literal as a codelist"

        inline_call = calls[0]
        assert inline_call[0] == "<inline>", inline_call
        assert any(part.startswith("source=codelists.py:") for part in inline_call[1:])
        assert any(
            part.startswith("values=1240751000000100|1325171000000109|1325181000000106")
            for part in inline_call[1:]
        )


def test_codelist_via_case_when_with_helper_function():
    """Test extracting codelists from case/when expressions with helper function calls.

    This pattern is used in JAKi-hosp where:
    1. Codelists are defined in a separate codelists.py file and imported with `from codelists import *`
    2. prostate_cancer_death = cause_of_death_matches(prostate_cancer_icd10)
    3. The helper function cause_of_death_matches takes a codelist parameter
    4. The result is used in a case/when expression in qa_bin_prostate_cancer

    The extractor should trace through function calls to find codelist parameters.
    """
    dataset_code = """
from ehrql import create_dataset, case, when
from ehrql.tables.tpp import clinical_events, apcs, ons_deaths
from codelists import *
from variable_helper_functions import *

dataset = create_dataset()

# Intermediate variables (not dataset variables)
prostate_cancer_snomed = clinical_events.where(
    clinical_events.snomedct_code.is_in(prostate_cancer_snomed_clinical)
).exists_for_patient()

prostate_cancer_hes = apcs.where(
    apcs.all_diagnoses.contains_any_of(prostate_cancer_icd10)
).exists_for_patient()

prostate_cancer_death = cause_of_death_matches(prostate_cancer_icd10)

# Final dataset variable using case/when with intermediate variables including helper function result
dataset.qa_bin_prostate_cancer = case(
    when(prostate_cancer_snomed).then(True),
    when(prostate_cancer_hes).then(True),
    when(prostate_cancer_death).then(True),
    otherwise=False
)
"""

    codelists_code = """
from ehrql import codelist_from_csv

prostate_cancer_snomed_clinical = codelist_from_csv("codelists/prostate_cancer_snomed.csv", column="code")
prostate_cancer_icd10 = codelist_from_csv("codelists/prostate_cancer_icd10.csv", column="code")
"""

    helper_code = """
from ehrql.tables.tpp import ons_deaths
import operator
from functools import reduce

def any_of(conditions):
    return reduce(operator.or_, conditions)

def cause_of_death_matches(codelist):
    conditions = [
        getattr(ons_deaths, column_name).is_in(codelist)
        for column_name in (["underlying_cause_of_death"]+[f"cause_of_death_{i:02d}" for i in range(1, 16)])
    ]
    return any_of(conditions)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = pathlib.Path(tmpdir)
        file_path = repo_root / "dataset_definition.py"
        file_path.write_text(dataset_code)

        codelists_path = repo_root / "codelists.py"
        codelists_path.write_text(codelists_code)

        helper_path = repo_root / "variable_helper_functions.py"
        helper_path.write_text(helper_code)

        codelists_result = extract_variable_codelists(file_path, repo_root)

        # qa_bin_prostate_cancer should find both codelists via intermediate variables
        assert "qa_bin_prostate_cancer" in codelists_result
        calls = codelists_result["qa_bin_prostate_cancer"]
        assert len(calls) >= 2, (
            f"Expected at least 2 codelist calls, got {len(calls)}: {calls}"
        )

        # Should find prostate_cancer_snomed.csv (appears once via prostate_cancer_snomed)
        snomed_calls = [c for c in calls if "prostate_cancer_snomed.csv" in c[0]]
        assert len(snomed_calls) == 1, (
            f"Should find prostate_cancer_snomed.csv once, got {calls}"
        )

        # Should find prostate_cancer_icd10.csv (appears twice: via prostate_cancer_hes and prostate_cancer_death)
        icd10_calls = [c for c in calls if "prostate_cancer_icd10.csv" in c[0]]
        # We expect at least ONE reference to prostate_cancer_icd10.csv
        assert len(icd10_calls) >= 1, (
            f"Should find prostate_cancer_icd10.csv at least once, got {calls}"
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
