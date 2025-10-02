import json

from dotenv import load_dotenv

from github_utils import RepoGetter
from parsing import get_study_variables


load_dotenv()


def main():
    results = {}
    parsing_errors = {}

    rg = RepoGetter()
    for repository, study_definitions in rg.get_all_cohortextractor_study_definitions():
        study_variables = {}
        study_errors = {}
        for study_name, contents in study_definitions:
            variables, errors = get_study_variables(contents)
            if variables:
                study_variables[study_name] = variables
            if errors:
                study_errors[study_name] = errors
        if study_variables:
            results[repository.name] = study_variables
        if study_errors:
            parsing_errors[repository.name] = study_errors

    with open("cohort_extractor_variables.json", "w") as f:
        json.dump(results, f, indent=4)

    with open("repository_errors.json", "w") as f:
        json.dump(rg.get_formatted_exceptions(), f, indent=4)

    with open("parsing_errors.json", "w") as f:
        json.dump(parsing_errors, f, indent=4)


if __name__ == "__main__":
    main()
