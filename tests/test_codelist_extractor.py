"""Test for ehrql_codelist_extractor.py"""

import pathlib
import tempfile

from parsing.ehrql_variable_extractor import (
    VariableExtractor,
    extract_variable_codelists,
)


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


def test_inline_codelist_with_star_import_attribute():
    """Star imports that refer to codelists.<name> should still resolve inline lists."""

    dataset_code = """
from ehrql import create_dataset
from codelists import *

dataset = create_dataset()

dataset.covid_critcare_date = records.where(
    records.code.is_in(codelists.covid_critcare_codes)
).first_for_patient().date
"""

    codelists_code = """
from ehrql import codelist_from_csv

covid_critcare_codes = codelist_from_csv(
    "codelists/covid-critical-care.csv",
    column="code",
)

# Override with inline tweak after review
covid_critcare_codes = [
    "U071",
    "U072",
    "U109",
]
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = pathlib.Path(tmpdir)
        file_path = repo_root / "dataset_definition.py"
        file_path.write_text(dataset_code)

        codelists_path = repo_root / "codelists.py"
        codelists_path.write_text(codelists_code)

        codelists = extract_variable_codelists(file_path, repo_root)

        assert "covid_critcare_date" in codelists
        calls = codelists["covid_critcare_date"]
        assert calls, "Should detect inline override despite star import"

        inline_call = calls[-1]
        assert inline_call[0] == "<inline>", inline_call
        assert any(part.startswith("values=U071|U072|U109") for part in inline_call[1:])


def test_parse_codelists_json_url_mapping():
    """Test that codelists.json is parsed correctly to get URL mapping."""
    import json

    from ehrql_codelist_extractor import normalize_path, parse_codelists_json

    codelists_json_content = {
        "files": {
            "asthma.csv": {
                "id": "opensafely/asthma/v1",
                "url": "https://codelists.opensafely.org/codelist/opensafely/asthma/v1/",
                "downloaded_at": "2023-01-01 12:00:00",
                "sha": "abc123",
            },
            "diabetes.csv": {
                "id": "opensafely/diabetes/v2",
                "url": "https://codelists.opensafely.org/codelist/opensafely/diabetes/v2/",
                "downloaded_at": "2023-01-01 12:00:00",
                "sha": "def456",
            },
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = pathlib.Path(tmpdir)
        codelists_dir = repo_root / "codelists"
        codelists_dir.mkdir()

        codelists_json = codelists_dir / "codelists.json"
        codelists_json.write_text(json.dumps(codelists_json_content))

        url_map, invalid_slugs = parse_codelists_json(repo_root)

        # No invalid slugs expected for these canonical URLs
        assert invalid_slugs == []

        # Should map both with and without "codelists/" prefix to slug form
        assert normalize_path("codelists/asthma.csv") in url_map
        assert (
            url_map[normalize_path("codelists/asthma.csv")] == "/opensafely/asthma/v1/"
        )

        assert normalize_path("asthma.csv") in url_map
        assert url_map[normalize_path("asthma.csv")] == "/opensafely/asthma/v1/"

        assert normalize_path("codelists/diabetes.csv") in url_map
        assert (
            url_map[normalize_path("codelists/diabetes.csv")]
            == "/opensafely/diabetes/v2/"
        )


def test_nested_dict_items_range_loop():
    """Test codelist extraction from nested for loops with dict.items() and range().

    This matches the pattern from pharmacy-first-data-development where:
    - Outer loop iterates over dict.items() where values reference codelist-using variables
    - Inner loop iterates over range(N)
    - Variables are added with f-strings containing both loop variables:
        dataset.add_column(f"{desc}_status{status}", count_query)

    See: pharmacy-first-data-development/analysis/dataset_definition_med_status_data_development.py
    """
    code = """
from ehrql import create_dataset, codelist_from_csv
from ehrql.tables.tpp import medications

# Define codelist
pharmacy_first_med_codes = codelist_from_csv(
    "codelists/pharmacy_first_meds.csv", column="code"
)

dataset = create_dataset()

# Pre-launch medication selection that uses the codelist
selected_medications_pre = (
    medications.where(
        medications.date.is_on_or_before("2024-01-01")
    )
    .where(medications.dmd_code.is_in(pharmacy_first_med_codes))
    .sort_by(medications.date)
)

# Post-launch medication selection that uses the codelist
selected_medications_post = (
    medications.where(
        medications.date.is_on_or_after("2024-02-01")
    )
    .where(medications.dmd_code.is_in(pharmacy_first_med_codes))
    .sort_by(medications.date)
)

# Dict mapping names to codelist-using query objects
selected_medications_dict = {
    "pre": selected_medications_pre,
    "post": selected_medications_post,
}

# Nested loop pattern: iterate dict.items() and range()
for desc, selected_medications in selected_medications_dict.items():
    for status in range(3):
        count_query = selected_medications.where(
            selected_medications.medication_status == status
        ).count_for_patient()
        dataset.add_column(f"{desc}_status{status}", count_query)
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = pathlib.Path(tmpdir)
        file_path = repo_root / "dataset_definition.py"
        file_path.write_text(code)

        # Use extract_codelist_calls_alt which supports nested dict.items()/range() loops
        extractor = VariableExtractor(file_path, repo_root)
        codelists = extractor.extract_codelist_calls_alt()

        # Should extract 6 variables: pre_status0, pre_status1, pre_status2,
        #                             post_status0, post_status1, post_status2
        expected_vars = [
            "pre_status0",
            "pre_status1",
            "pre_status2",
            "post_status0",
            "post_status1",
            "post_status2",
        ]

        for var in expected_vars:
            assert var in codelists, f"Variable {var} should be extracted"
            assert len(codelists[var]) >= 1, f"Variable {var} should have codelist call"
            codelist_files = [call[0] for call in codelists[var]]
            assert "codelists/pharmacy_first_meds.csv" in codelist_files, (
                f"Variable {var} should reference pharmacy_first_meds.csv"
            )


def test_nested_loop_with_getattr_codelist():
    """Test codelist extraction from nested loops using getattr to access codelists.

    This matches the pattern from inflammatory_rheum/analysis/dataset_definition_incidence.py where:
    - Variables are created in nested loops over disease names and codelist types
    - Codelists are accessed dynamically via getattr(codelists, f"{disease}_snomed")
    - The codelist is passed to a helper function whose result is used in add_column

    Example pattern:
        for disease in diseases:
            if hasattr(codelists, f"{disease}_snomed"):
                disease_codelist = getattr(codelists, f"{disease}_snomed")
                dataset.add_column(f"{disease}_prim_count", count_func(disease_codelist))
    """
    # Main dataset definition file
    code = """
from ehrql import create_dataset
from ehrql.tables.tpp import clinical_events
import codelists

diseases = ["asthma", "copd"]

dataset = create_dataset()

def count_codes(dx_codelist):
    return clinical_events.where(
        clinical_events.snomedct_code.is_in(dx_codelist)
    ).count_for_patient()

for disease in diseases:
    if hasattr(codelists, f"{disease}_snomed"):
        disease_codelist = getattr(codelists, f"{disease}_snomed")
        dataset.add_column(f"{disease}_count", count_codes(disease_codelist))
"""

    # Codelists module with actual codelist definitions
    codelists_code = """
from ehrql import codelist_from_csv

asthma_snomed = codelist_from_csv("codelists/asthma.csv", column="code")
copd_snomed = codelist_from_csv("codelists/copd.csv", column="code")
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = pathlib.Path(tmpdir)
        file_path = repo_root / "dataset_definition.py"
        file_path.write_text(code)

        # Create codelists.py module
        codelists_path = repo_root / "codelists.py"
        codelists_path.write_text(codelists_code)

        # Use extract_codelist_calls_alt
        extractor = VariableExtractor(file_path, repo_root)
        codelists_result = extractor.extract_codelist_calls_alt()

        # Should extract asthma_count and copd_count variables
        expected_vars = {
            "asthma_count": "codelists/asthma.csv",
            "copd_count": "codelists/copd.csv",
        }

        for var, expected_codelist in expected_vars.items():
            assert var in codelists_result, f"Variable {var} should be extracted"
            assert len(codelists_result[var]) >= 1, (
                f"Variable {var} should have codelist call"
            )
            codelist_files = [call[0] for call in codelists_result[var]]
            assert expected_codelist in codelist_files, (
                f"Variable {var} should reference {expected_codelist}, got {codelist_files}"
            )


def test_getattr_codelist_in_creator_function():
    """Test codelist extraction when getattr loop is inside a dataset creator function.

    This matches the pattern from inflammatory_rheum/analysis/dataset_definition_incidence.py where:
    - A function creates the dataset and adds columns in a loop
    - The function is called at module level: dataset = create_dataset_with_variables()
    - Inside the function, codelists are accessed via getattr(codelists, f"{disease}_snomed")

    Example pattern:
        def create_dataset_with_variables():
            dataset = create_dataset()
            for disease in diseases:
                if hasattr(codelists, f"{disease}_snomed"):
                    disease_codelist = getattr(codelists, f"{disease}_snomed")
                    dataset.add_column(f"{disease}_count", count_func(disease_codelist))
            return dataset

        dataset = create_dataset_with_variables()
    """
    # Main dataset definition file
    code = """
from ehrql import create_dataset
from ehrql.tables.tpp import clinical_events
import codelists

diseases = ["asthma", "copd"]

def create_dataset_with_variables():
    dataset = create_dataset()

    def count_codes(dx_codelist):
        return clinical_events.where(
            clinical_events.snomedct_code.is_in(dx_codelist)
        ).count_for_patient()

    for disease in diseases:
        if hasattr(codelists, f"{disease}_snomed"):
            disease_codelist = getattr(codelists, f"{disease}_snomed")
            dataset.add_column(f"{disease}_count", count_codes(disease_codelist))

    return dataset

dataset = create_dataset_with_variables()
"""

    # Codelists module with actual codelist definitions
    codelists_code = """
from ehrql import codelist_from_csv

asthma_snomed = codelist_from_csv("codelists/asthma.csv", column="code")
copd_snomed = codelist_from_csv("codelists/copd.csv", column="code")
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = pathlib.Path(tmpdir)
        file_path = repo_root / "dataset_definition.py"
        file_path.write_text(code)

        # Create codelists.py module
        codelists_path = repo_root / "codelists.py"
        codelists_path.write_text(codelists_code)

        # Use extract_codelist_calls_alt
        extractor = VariableExtractor(file_path, repo_root)
        codelists_result = extractor.extract_codelist_calls_alt()

        # Should extract asthma_count and copd_count variables
        expected_vars = {
            "asthma_count": "codelists/asthma.csv",
            "copd_count": "codelists/copd.csv",
        }

        for var, expected_codelist in expected_vars.items():
            assert var in codelists_result, f"Variable {var} should be extracted"
            assert len(codelists_result[var]) >= 1, (
                f"Variable {var} should have codelist call"
            )
            codelist_files = [call[0] for call in codelists_result[var]]
            assert expected_codelist in codelist_files, (
                f"Variable {var} should reference {expected_codelist}, got {codelist_files}"
            )


def test_nested_loop_with_if_branches_and_getattr():
    """Test codelist extraction for nested loops with if/elif branches.

    This matches the pattern from disease_incidence/analysis/dataset_definition_demographics_disease.py:
    - Two nested loops: for disease in diseases, for codelist_type in codelist_types
    - if/elif conditions check codelist_type to determine which codelist suffix to use
    - hasattr check before getattr to handle missing codelists
    - Different column name suffixes for each branch (_prim_date vs _sec_date)

    The key issue being fixed: when multiple if/elif branches assign to the same variable
    name (e.g., disease_codelist), each add_column call should use the getattr from its
    own branch, not from a different branch.
    """
    # Main dataset definition file
    code = """
from ehrql import create_dataset
from ehrql.tables.tpp import clinical_events, apcs
import codelists

diseases = ["asthma", "copd"]
codelist_types = ["snomed", "icd"]

dataset = create_dataset()

for disease in diseases:
    for codelist_type in codelist_types:
        if codelist_type == "snomed":
            if hasattr(codelists, f"{disease}_snomed"):
                disease_codelist = getattr(codelists, f"{disease}_snomed")
                dataset.add_column(f"{disease}_prim_date", clinical_events.where(
                    clinical_events.snomedct_code.is_in(disease_codelist)
                ).first_for_patient().date)
        elif codelist_type == "icd":
            if hasattr(codelists, f"{disease}_icd"):
                disease_codelist = getattr(codelists, f"{disease}_icd")
                dataset.add_column(f"{disease}_sec_date", apcs.where(
                    apcs.primary_diagnosis.is_in(disease_codelist)
                ).first_for_patient().admission_date)
"""

    # Codelists module with actual codelist definitions
    codelists_code = """
from ehrql import codelist_from_csv

asthma_snomed = codelist_from_csv("codelists/asthma_snomed.csv", column="code")
asthma_icd = codelist_from_csv("codelists/asthma_icd.csv", column="code")
copd_snomed = codelist_from_csv("codelists/copd_snomed.csv", column="code")
copd_icd = codelist_from_csv("codelists/copd_icd.csv", column="code")
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = pathlib.Path(tmpdir)
        file_path = repo_root / "dataset_definition.py"
        file_path.write_text(code)

        # Create codelists.py module
        codelists_path = repo_root / "codelists.py"
        codelists_path.write_text(codelists_code)

        # Use extract_codelist_calls_alt
        extractor = VariableExtractor(file_path, repo_root)
        codelists_result = extractor.extract_codelist_calls_alt()

        # Should extract *_prim_date and *_sec_date variables with their correct codelists
        # The key test: prim_date should have snomed codelist, sec_date should have icd codelist
        expected_vars = {
            "asthma_prim_date": "codelists/asthma_snomed.csv",
            "asthma_sec_date": "codelists/asthma_icd.csv",
            "copd_prim_date": "codelists/copd_snomed.csv",
            "copd_sec_date": "codelists/copd_icd.csv",
        }

        for var, expected_codelist in expected_vars.items():
            assert var in codelists_result, f"Variable {var} should be extracted"
            assert len(codelists_result[var]) >= 1, (
                f"Variable {var} should have codelist call"
            )
            codelist_files = [call[0] for call in codelists_result[var]]
            assert expected_codelist in codelist_files, (
                f"Variable {var} should reference {expected_codelist}, got {codelist_files}"
            )


def test_local_dict_intermediate_variables():
    """Test codelist extraction when codelists are stored in local dict variables.

    This matches the pattern from disease_incidence/analysis/dataset_definition.py:
    - Codelists are stored in local dict variables (snomed_inc_date, icd_inc_date)
    - Then add_column uses minimum_of() referencing these dict values
    - Later variables use getattr(dataset, ...) to reference the inc_date

    Pattern:
        snomed_inc_date = {}
        icd_inc_date = {}
        for disease in diseases:
            if hasattr(codelists, f"{disease}_snomed"):
                disease_codelist = getattr(codelists, f"{disease}_snomed")
                snomed_inc_date[f"{disease}_snomed_inc_date"] = first_code(..., disease_codelist).date
            if hasattr(codelists, f"{disease}_icd"):
                disease_codelist = getattr(codelists, f"{disease}_icd")
                icd_inc_date[f"{disease}_icd_inc_date"] = first_code(..., disease_codelist).date
            dataset.add_column(f"{disease}_inc_date",
                minimum_of(snomed_inc_date[f"..."], icd_inc_date[f"..."]))
            dataset.add_column(f"{disease}_alive_inc",
                dataset.date_of_death.is_after(getattr(dataset, f"{disease}_inc_date")))
    """
    # Main dataset definition file
    code = """
from ehrql import create_dataset, minimum_of
from ehrql.tables.tpp import clinical_events, apcs, patients
import codelists

diseases = ["dementia"]

dataset = create_dataset()
dataset.date_of_death = patients.date_of_death

def first_code_snomed(codelist):
    return clinical_events.where(
        clinical_events.snomedct_code.is_in(codelist)
    ).sort_by(clinical_events.date).first_for_patient()

def first_code_icd(codelist):
    return apcs.where(
        apcs.primary_diagnosis.is_in(codelist)
    ).sort_by(apcs.admission_date).first_for_patient()

for disease in diseases:
    snomed_inc_date = {}
    icd_inc_date = {}

    if hasattr(codelists, f"{disease}_snomed"):
        disease_codelist = getattr(codelists, f"{disease}_snomed")
        snomed_inc_date[f"{disease}_snomed_inc_date"] = first_code_snomed(disease_codelist).date
    else:
        snomed_inc_date[f"{disease}_snomed_inc_date"] = None

    if hasattr(codelists, f"{disease}_icd"):
        disease_codelist = getattr(codelists, f"{disease}_icd")
        icd_inc_date[f"{disease}_icd_inc_date"] = first_code_icd(disease_codelist).admission_date
    else:
        icd_inc_date[f"{disease}_icd_inc_date"] = None

    dataset.add_column(f"{disease}_inc_date",
        minimum_of(*[date for date in [
            snomed_inc_date[f"{disease}_snomed_inc_date"],
            icd_inc_date[f"{disease}_icd_inc_date"]
        ] if date is not None])
    )

    dataset.add_column(f"{disease}_alive_inc",
        (
            (dataset.date_of_death.is_after(getattr(dataset, f"{disease}_inc_date"))) |
            dataset.date_of_death.is_null()
        ).when_null_then(False)
    )
"""

    # Codelists module with actual codelist definitions
    codelists_code = """
from ehrql import codelist_from_csv

dementia_snomed = codelist_from_csv("codelists/dementia_snomed.csv", column="code")
dementia_icd = codelist_from_csv("codelists/dementia_icd.csv", column="code")
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = pathlib.Path(tmpdir)
        file_path = repo_root / "dataset_definition.py"
        file_path.write_text(code)

        # Create codelists.py module
        codelists_path = repo_root / "codelists.py"
        codelists_path.write_text(codelists_code)

        # Use extract_codelist_calls_alt
        extractor = VariableExtractor(file_path, repo_root)
        codelists_result = extractor.extract_codelist_calls_alt()

        # dementia_inc_date should reference both codelists via the local dict variables
        assert "dementia_inc_date" in codelists_result, (
            f"Variable dementia_inc_date should be extracted, got: {list(codelists_result.keys())}"
        )
        assert len(codelists_result["dementia_inc_date"]) >= 2, (
            f"dementia_inc_date should have 2 codelist calls, got {len(codelists_result['dementia_inc_date'])}"
        )
        codelist_files = [call[0] for call in codelists_result["dementia_inc_date"]]
        assert "codelists/dementia_snomed.csv" in codelist_files, (
            f"dementia_inc_date should reference dementia_snomed.csv, got {codelist_files}"
        )
        assert "codelists/dementia_icd.csv" in codelist_files, (
            f"dementia_inc_date should reference dementia_icd.csv, got {codelist_files}"
        )

        # dementia_alive_inc references dementia_inc_date, so it should also have both codelists
        assert "dementia_alive_inc" in codelists_result, (
            f"Variable dementia_alive_inc should be extracted, got: {list(codelists_result.keys())}"
        )
        assert len(codelists_result["dementia_alive_inc"]) >= 2, (
            f"dementia_alive_inc should have 2 codelist calls, got {len(codelists_result['dementia_alive_inc'])}"
        )
        codelist_files = [call[0] for call in codelists_result["dementia_alive_inc"]]
        assert "codelists/dementia_snomed.csv" in codelist_files, (
            f"dementia_alive_inc should reference dementia_snomed.csv, got {codelist_files}"
        )
        assert "codelists/dementia_icd.csv" in codelist_files, (
            f"dementia_alive_inc should reference dementia_icd.csv, got {codelist_files}"
        )


def test_getattr_dataset_reference_inherits_codelists():
    """Test that variables referencing other dataset variables via getattr(dataset, ...) inherit codelists.

    This matches the pattern from disease_incidence/analysis/dataset_definition_demographics_disease.py:
        dataset.add_column(f"{disease}_inc_date",
            minimum_of(
                getattr(dataset, f"{disease}_prim_date", None),
                getattr(dataset, f"{disease}_sec_date", None)
            )
        )

    The ulcerative_colitis_inc_date variable should inherit codelists from
    ulcerative_colitis_prim_date and ulcerative_colitis_sec_date.
    """
    # Main dataset definition file
    code = """
from ehrql import create_dataset, minimum_of
from ehrql.tables.tpp import clinical_events, apcs
import codelists

diseases = ["asthma"]

dataset = create_dataset()

for disease in diseases:
    # First define the base variables with codelists
    disease_snomed = getattr(codelists, f"{disease}_snomed")
    dataset.add_column(f"{disease}_prim_date", clinical_events.where(
        clinical_events.snomedct_code.is_in(disease_snomed)
    ).first_for_patient().date)

    disease_icd = getattr(codelists, f"{disease}_icd")
    dataset.add_column(f"{disease}_sec_date", apcs.where(
        apcs.primary_diagnosis.is_in(disease_icd)
    ).first_for_patient().admission_date)

    # Now define a variable that references the other dataset variables
    dataset.add_column(f"{disease}_inc_date",
        minimum_of(
            getattr(dataset, f"{disease}_prim_date", None),
            getattr(dataset, f"{disease}_sec_date", None)
        )
    )
"""

    # Codelists module with actual codelist definitions
    codelists_code = """
from ehrql import codelist_from_csv

asthma_snomed = codelist_from_csv("codelists/asthma_snomed.csv", column="code")
asthma_icd = codelist_from_csv("codelists/asthma_icd.csv", column="code")
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = pathlib.Path(tmpdir)
        file_path = repo_root / "dataset_definition.py"
        file_path.write_text(code)

        # Create codelists.py module
        codelists_path = repo_root / "codelists.py"
        codelists_path.write_text(codelists_code)

        # Use extract_codelist_calls_alt
        extractor = VariableExtractor(file_path, repo_root)
        codelists_result = extractor.extract_codelist_calls_alt()

        # asthma_inc_date references asthma_prim_date (snomed) and asthma_sec_date (icd)
        # So it should have both codelists
        assert "asthma_inc_date" in codelists_result, (
            "Variable asthma_inc_date should be extracted"
        )
        assert len(codelists_result["asthma_inc_date"]) >= 2, (
            f"asthma_inc_date should have 2 codelist calls, got {len(codelists_result['asthma_inc_date'])}"
        )
        codelist_files = [call[0] for call in codelists_result["asthma_inc_date"]]
        assert "codelists/asthma_snomed.csv" in codelist_files, (
            f"asthma_inc_date should reference asthma_snomed.csv, got {codelist_files}"
        )
        assert "codelists/asthma_icd.csv" in codelist_files, (
            f"asthma_inc_date should reference asthma_icd.csv, got {codelist_files}"
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
