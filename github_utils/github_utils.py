import os
from collections.abc import Iterator, Mapping
from typing import Any, Optional

import yaml
from github import Auth, ContentFile, Github
from github.GithubException import GithubException
from github.Repository import Repository


def get_cohortextractor_study_definitions(
    organisation: str = "opensafely",
) -> Iterator[tuple[Repository, list[tuple[str, str]]]]:
    organisation_client = _get_organisation_client(organisation)

    for repository in organisation_client.get_repos():
        actions = _get_cohortextractor_actions(repository=repository)
        if not actions:
            continue
        study_definitions_lists = [
            _get_study_definitions(repository=repository, action=action)
            for action in actions
        ]
        study_definitions = [
            study_definition
            for study_definition_list in study_definitions_lists
            for study_definition in study_definition_list
        ]

        yield (
            repository,
            [
                (
                    study_definition.name,
                    study_definition.decoded_content.decode("utf-8"),
                )
                for study_definition in study_definitions
            ],
        )


def _get_cohortextractor_actions(repository: Repository) -> Optional[list[Any]]:
    project_config = _get_project_config(repository=repository)
    if not project_config:
        return

    actions_section = project_config.get("actions")
    if not isinstance(actions_section, Mapping):
        return

    return [
        action
        for action in actions_section.values()
        if "run" in action and "cohortextractor" in action["run"]
    ]


def _get_study_definitions(
    repository: Repository, action: Mapping
) -> list[ContentFile.ContentFile]:
    def infer_paths(action):
        output_paths = action["outputs"]["highly_sensitive"].values()
        return [
            output_path.replace("output/", "analysis/")
            .replace("input_", "study_definition_")
            .replace(".csv", ".py")
            for output_path in output_paths
        ]

    def fetch_path(path):
        try:
            study_definition = repository.get_contents(path)
            if isinstance(study_definition, list):
                study_definition = study_definition[0]
            return study_definition
        except GithubException as exc:
            if exc.status == 404:
                raise ValueError(f"Unable to locate {path} in {repository.name}")
            else:
                raise exc

    run_stmt = action["run"]
    if "--study-definition" in run_stmt:
        run_split = run_stmt.split(" ")
        path = run_split[run_split.index("--study-definition") + 1]
    else:
        path = "study_definition"
    path = f"analysis/{path}.py"
    try:
        study_definition = fetch_path(path)
        study_definitions = [study_definition]
    except ValueError as exc:
        if f"Unable to locate {path}" in str(exc):
            # If there's an explicit path and we can't find it, don't try to infer, just move on
            if "--study-definition" in run_stmt:
                return []
            # we've got a cohort extractor action defined
            # and no explicit or default study definition path.
            # Must be using cohortextractor's quirky path inference,
            # let's do our best.
            candidate_paths = infer_paths(action=action)
            study_definitions = []
            for path in candidate_paths:
                try:
                    study_definition = fetch_path(path)
                    study_definitions.append(study_definition)
                except ValueError as inexc:
                    if f"Unable to locate {path}" in str(inexc):
                        continue
                    else:
                        raise inexc
            if not study_definitions:
                raise ValueError(
                    f"Unable to find any study definitions for {action['run']} in {repository.name}"
                )
        else:
            raise exc
    return study_definitions


def _get_project_config(repository: Repository):
    try:
        project_file = repository.get_contents("project.yaml")
    except GithubException as exc:  # pragma: no cover - network errors in prod only
        if exc.status == 404:
            return
        raise
    if isinstance(project_file, list):
        project_file = project_file[0]
    project_yaml = project_file.decoded_content.decode("utf-8")

    try:
        return yaml.safe_load(project_yaml) or {}
    # "research_temp_tm" has an invalid project.yaml, there may be others!
    except (
        yaml.YAMLError,
        yaml.parser.ParserError,
    ):
        return {}


def _get_organisation_client(organisation):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        msg = "GITHUB_TOKEN environment variable must be set to call GitHub APIs"
        raise RuntimeError(msg)
    github_client = Github(auth=Auth.Token(token))
    organisation_client = github_client.get_organization(organisation)
    return organisation_client
