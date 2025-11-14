import marimo


__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import csv
    import json
    from collections import Counter, defaultdict
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    return Counter, Path, csv, datetime, defaultdict, json, mo, pd


@app.cell
def _(Path, csv):
    # load job server data file, restructure for convenience/performance
    jobs = list(
        csv.DictReader(Path("data/All.jobs-data-2025-11-10.18_25_26.csv").open())
    )
    job_shas = {job["sha"] for job in jobs}
    successful_job_shas = {job["sha"] for job in jobs if job["_status"] == "succeeded"}
    return job_shas, jobs, successful_job_shas


@app.cell
def _(Path, json):
    # load OpenCodelists data file
    codelists = json.load(Path("data/rsi-codelists-analysis.json").open())
    return (codelists,)


@app.cell
def _(Path, defaultdict, json):
    # load ehrl codelist usage file, restructure for convenience/performance
    ehrql_codelists = json.load(Path("data/ehrql_codelists.json").open())
    signatures_to_shas = defaultdict(set[str])
    signatures_to_projects = defaultdict(str)
    for _project, _hashes in ehrql_codelists["projects"].items():
        for _sha, _signature in _hashes.items():
            signatures_to_shas[_signature].add(_sha)
            signatures_to_projects[_signature] = _project
    return ehrql_codelists, signatures_to_projects, signatures_to_shas


@app.cell
def _(mo):
    mo.md("""## Overall codelist counts on OpenCodelists""")
    return


@app.cell
def _(codelists, mo):
    mo.md(
        f"""Total number of codelist with at least one under review/published version: {len(codelists)}"""
    )
    return


@app.cell
def _(codelists, mo):
    mo.md(
        f"""Total number of published/under review codelist versions: {sum([len(codelist["versions"]) for codelist in codelists])}"""
    )
    return


@app.cell
def _(codelists, pd):
    df_versioncount = pd.DataFrame(
        [
            {_k: _v for _k, _v in _codelist.items() if _k in ["owner", "coding_system"]}
            | {"version_count": len(_codelist["versions"])}
            for _codelist in codelists
        ]
    )
    return (df_versioncount,)


@app.cell
def _(df_versioncount):
    df_versioncount.groupby("owner").sum("version_count")
    return


@app.cell
def _(df_versioncount):
    df_versioncount.groupby("coding_system").sum("version_count")
    return


@app.cell
def _(
    defaultdict,
    ehrql_codelists,
    job_shas,
    signatures_to_projects,
    signatures_to_shas,
):
    # link ehrl codelist data and job server data to count codelist usage by job
    codelist_projects = defaultdict(set[str])
    codelist_variables = defaultdict(int)
    codelist_jobs = defaultdict(set[str])
    for _signature, _files in ehrql_codelists["signatures"].items():
        _project = signatures_to_projects[_signature]
        _shas = {_sha for _sha in signatures_to_shas[_signature] if _sha in job_shas}
        for _file, _variables in _files.items():
            for _name, _codelists in _variables.items():
                for _definition in _codelists:
                    if len(_definition) == 2:
                        _codelist = _definition[0]
                        codelist_variables[_codelist] += 1
                        codelist_projects[_codelist].add(_project)
                        codelist_jobs[_codelist] |= _shas
    return codelist_jobs, codelist_projects, codelist_variables


@app.cell
def _(mo):
    mo.md("""## Codelist versions in any job:""")
    return


@app.cell
def _(codelist_jobs, pd):
    pd.DataFrame(
        [
            {
                "codelist_version_slug": _k,
                "count_variables_X_jobs_using_codelist": len(_v),
            }
            for _k, _v in codelist_jobs.items()
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""## Codelist versions in successful jobs:""")
    return


@app.cell
def _(codelist_jobs, pd, successful_job_shas):
    pd.DataFrame(
        [
            {
                "codelist_version_slug": _k,
                "count_variables_X_successful_jobs_using_codelist": len(
                    _v.intersection(successful_job_shas)
                ),
            }
            for _k, _v in codelist_jobs.items()
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""## Numbers of variables featuring codelist version:""")
    return


@app.cell
def _(codelist_variables, pd):
    pd.DataFrame(
        [
            {"codelist_version_slug": _k, "count_of_variables_using_codelist": _v}
            for _k, _v in codelist_variables.items()
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""## Numbers of projects featuring codelist version:""")
    return


@app.cell
def _(codelist_projects, pd):
    pd.DataFrame(
        [
            {
                "codelist_version_slug": _k,
                "count_projects_with_variable_using_codelist": len(_v),
            }
            for _k, _v in codelist_projects.items()
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Stray observervation: some slugs don't have version identifiers?""")
    return


@app.function
def has_version_identifier(_codelist: str) -> bool | None:
    if _codelist:
        _parts = _codelist.strip("/").split("/")
        return len(_parts) == (4 if _parts[0] == "user" else 3)


@app.cell
def _(Counter, codelist_variables):
    # how many are referred to with a version identifier
    Counter(
        [has_version_identifier(_codelist) for _codelist in codelist_variables.keys()]
    )
    return


@app.cell
def _(codelist_variables):
    # local codelists/failure to match up in codelists.txt?
    [
        _codelist
        for _codelist in codelist_variables.keys()
        if not has_version_identifier(_codelist)
    ]
    return


@app.cell
def _(codelist_jobs, codelists, mo, pd):
    _ever_used_slugs = set(_k.lstrip("/") for _k in codelist_jobs.keys() if _k)
    _never_used = []
    for _codelist in codelists:
        for _version in _codelist["versions"]:
            if _version["slug"] in _ever_used_slugs:
                continue
            _never_used.append(
                {
                    _k: _v
                    for _k, _v in _codelist.items()
                    if _k in ["name", "owner", "coding_system"]
                }
                | {
                    _k: _v
                    for _k, _v in _version.items()
                    if _k in ["slug", "updated_at", "status", "creation_method"]
                }
            )
    never_used = pd.DataFrame(_never_used)
    mo.md("## Never-used codelists:")
    return (never_used,)


@app.cell
def _(never_used):
    never_used
    return


@app.cell
def _(never_used):
    never_used.groupby("coding_system").count()["name"].rename(
        "never_used_codelistversion_count"
    )
    return


@app.cell
def _(never_used):
    never_used.groupby("owner").count()["name"].rename(
        "never_used_codelistversion_count"
    )
    return


@app.cell
def _(codelist_jobs, codelists, datetime, jobs, pd):
    # calculate time delta between codelist version update and job execution
    _sucessful_job_dates = {
        _job["sha"]: datetime.fromisoformat(_job["created_at"]).replace(tzinfo=None)
        for _job in jobs
    }
    _version_updated_at = {}
    for _codelist in codelists:
        for _version in _codelist["versions"]:
            _version_updated_at[_version["slug"]] = datetime.fromisoformat(
                _version["updated_at"]
            ).replace(tzinfo=None)
            if _version["tag"] and _version["slug"].endswith(_version["tag"]):
                _version_updated_at[
                    _version["slug"].replace(_version["tag"], _version["hash"])
                ] = datetime.fromisoformat(_version["updated_at"]).replace(tzinfo=None)
    _codelist_deltas = {
        _codelist_slug.strip("/"): {
            _sucessful_job_dates[_sha] - _version_updated_at[_codelist_slug.strip("/")]
            for _sha in _shas
        }
        for _codelist_slug, _shas in codelist_jobs.items()
        if _codelist_slug and _codelist_slug.strip("/") in _version_updated_at
    }

    df_timedeltas = pd.DataFrame(
        [
            {
                "slug": _c["owner"],
                "coding_system": _c["coding_system"],
                "updated_at": _v["updated_at"],
                "delta": (
                    _codelist_deltas.get(_v["slug"])
                    or _codelist_deltas.get(_c["slug"] + "/" + _v["hash"])
                ),
            }
            for _c in codelists
            for _v in _c["versions"]
        ]
    )
    return (df_timedeltas,)


@app.cell
def _(df_timedeltas):
    df_timedeltas
    return


if __name__ == "__main__":
    app.run()
