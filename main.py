import json

from dotenv import load_dotenv

from github_utils import get_cohortextractor_study_definitions
from parsing import get_study_variables


load_dotenv()


def main():
    results = {
        repository.name: [
            {study_name: get_study_variables(contents)}
            for study_name, contents in study_definitions
        ]
        for repository, study_definitions in get_cohortextractor_study_definitions()
    }

    with open("cohort_extractor_variables.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
