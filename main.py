import argparse
import json
import subprocess
import sys

from dotenv import load_dotenv

from github_utils import RepoGetter
from parsing import get_study_variables


load_dotenv()


def fetch():
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


def _launch_notebook(notebook_path: str | None) -> None:
    """Start marimo in edit mode for the given notebook."""

    command = [sys.executable, "-m", "marimo", "edit"] + (
        [notebook_path] if notebook_path else []
    )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cohort extractor variable utilities")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("fetch", help="Fetch cohort extractor variables and errors")

    notebook_parser = subparsers.add_parser(
        "notebook", help="Open the marimo in edit mode"
    )
    notebook_parser.add_argument(
        "--path",
        help="Path to the marimo notebook to edit",
    )

    args = parser.parse_args()

    if "command" in args:
        if args.command == "fetch":
            fetch()
        elif args.command == "notebook":
            _launch_notebook(args.path)
    parser.print_help()
    sys.exit(1)
