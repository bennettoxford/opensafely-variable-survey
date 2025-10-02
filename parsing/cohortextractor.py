import ast
from traceback import format_exception
from typing import Optional


def get_study_variables(
    study_definition: str,
) -> tuple[list[tuple[str, str]], list[str]]:
    try:
        study_definition_call = _parse_study_definition(
            study_definition=study_definition
        )
        if study_definition_call:
            return (
                _extract_variables(study_definition_call=study_definition_call),
                [],
            )
        else:
            return ([], [])
    except Exception as e:
        return ([], format_exception(e))


def _parse_study_definition(study_definition: str) -> Optional[ast.Call]:
    tree = ast.parse(study_definition)
    for stmt in [stmt for stmt in tree.body if isinstance(stmt, ast.Assign)]:
        if (
            isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "StudyDefinition"
        ):
            return stmt.value


def _extract_variables(study_definition_call: ast.Call) -> list[tuple[str, str]]:
    non_variable_args = ["default_expectations", "population"]
    return [
        (kw.arg, ast.unparse(kw.value))
        for kw in study_definition_call.keywords
        if kw.arg and kw.arg not in non_variable_args
    ]
