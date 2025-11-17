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
    import matplotlib.pyplot as plot
    import numpy as np
    import pandas as pd

    return Counter, Path, csv, datetime, defaultdict, json, mo, np, pd, plot


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
                "count_projects_with_variable_using_codelist_version": len(_v),
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
    ever_used_slugs = set(_k.strip("/") for _k in codelist_jobs.keys() if _k)
    _never_used = []
    for _codelist in codelists:
        for _version in _codelist["versions"]:
            if _version["slug"] in ever_used_slugs:
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
    mo.md("## Never-used codelist versions:")
    return ever_used_slugs, never_used


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
def _(codelists, ever_used_slugs, mo, pd):
    _never_used = []
    for _codelist in codelists:
        if any(e for e in ever_used_slugs if e.startswith(_codelist["slug"])):
            continue
        _never_used.append(
            {
                _k: _v
                for _k, _v in _codelist.items()
                if _k in ["name", "owner", "coding_system"]
            }
        )
    never_used_codelists = pd.DataFrame(_never_used)
    mo.md("## Never-used codelists:")
    return (never_used_codelists,)


@app.cell
def _(never_used_codelists):
    never_used_codelists
    return


@app.cell
def _(never_used_codelists):
    never_used_codelists.groupby("coding_system").count()["name"].rename(
        "never_used_codelist_count"
    )
    return


@app.cell
def _(never_used_codelists):
    never_used_codelists.groupby("owner").count()["name"].rename(
        "never_used_codelist_count"
    )
    return


@app.cell
def _(mo):
    mo.md("## Most popular codelists (across all versions)")
    return


@app.cell
def _(codelist_variables, defaultdict, pd):
    _codelist_variable_counts = defaultdict(int)
    for _k, _v in codelist_variables.items():
        if not _k:
            continue
        _slug_parts = _k.strip("/").split("/")
        _codelist_variable_counts["/".join(_slug_parts[: len(_slug_parts) - 1])] += _v
    pd.DataFrame(
        [
            {"codelist_slug": _k, "count_of_variables_using_codelist": _v}
            for _k, _v in _codelist_variable_counts.items()
        ]
    )
    return


@app.cell
def _(codelist_projects, defaultdict, pd):
    _codelist_project_counts = defaultdict(int)
    for _k, _v in codelist_projects.items():
        if not _k:
            continue
        _slug_parts = _k.strip("/").split("/")
        _codelist_project_counts["/".join(_slug_parts[: len(_slug_parts) - 1])] += len(
            _v
        )
    pd.DataFrame(
        [
            {
                "codelist_slug": _k,
                "count_projects_with_variable_using_codelist": _v,
            }
            for _k, _v in _codelist_project_counts.items()
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("## How out of date are codelists when used?")
    return


@app.cell
def _(codelist_jobs, codelists, datetime, jobs, pd):
    # calculate time delta between codelist version update and job execution
    sucessful_job_dates = {
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
            sucessful_job_dates[_sha] - _version_updated_at[_codelist_slug.strip("/")]
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
                    _codelist_deltas.get(_v["slug"], set())
                    | _codelist_deltas.get(_c["slug"] + "/" + _v["hash"], set())
                ),
            }
            for _c in codelists
            for _v in _c["versions"]
        ]
    )
    return df_timedeltas, sucessful_job_dates


@app.cell
def _(df_timedeltas):
    df_timedeltas
    return


@app.cell
def _(df_timedeltas):
    all_delta_days = [
        d.days
        for ds in df_timedeltas[df_timedeltas.delta.apply(len) > 0].delta.values
        for d in ds
    ]
    return (all_delta_days,)


@app.cell
def _(all_delta_days, plot):
    plot.hist(x=all_delta_days)[2][0]
    return


@app.cell
def _(mo):
    mo.md("""`updated_at` gets updated when the underlying coding system is updated (in most but not all circumstances) therefore unreliable!

    Let's re-run with `created_at` and accept there might be some errors where a Codelist Version is updated significantly after creation
    """)
    return


@app.cell
def _(codelist_jobs, codelists, datetime, pd, sucessful_job_dates):
    # calculate time delta between codelist version creation and job execution
    _version_created_at = {}
    for _codelist in codelists:
        for _version in _codelist["versions"]:
            _version_created_at[_version["slug"]] = datetime.fromisoformat(
                _version["created_at"]
            ).replace(tzinfo=None)
            if _version["tag"] and _version["slug"].endswith(_version["tag"]):
                _version_created_at[
                    _version["slug"].replace(_version["tag"], _version["hash"])
                ] = datetime.fromisoformat(_version["created_at"]).replace(tzinfo=None)
    _codelist_deltas = {
        _codelist_slug.strip("/"): {
            sucessful_job_dates[_sha] - _version_created_at[_codelist_slug.strip("/")]
            for _sha in _shas
        }
        for _codelist_slug, _shas in codelist_jobs.items()
        if _codelist_slug and _codelist_slug.strip("/") in _version_created_at
    }

    df_timedeltas_created = pd.DataFrame(
        [
            {
                "slug": _c["owner"],
                "coding_system": _c["coding_system"],
                "created_at": _v["created_at"],
                "delta": (
                    _codelist_deltas.get(_v["slug"], set())
                    | _codelist_deltas.get(_c["slug"] + "/" + _v["hash"], set())
                ),
            }
            for _c in codelists
            for _v in _c["versions"]
        ]
    )
    return (df_timedeltas_created,)


@app.cell
def _(df_timedeltas_created):
    all_delta_days_created = [
        d.days
        for ds in df_timedeltas_created[
            df_timedeltas_created.delta.apply(len) > 0
        ].delta.values
        for d in ds
    ]
    return (all_delta_days_created,)


@app.cell
def _(all_delta_days_created, plot):
    plot.hist(x=all_delta_days_created)[2][0]
    return


@app.cell
def _(all_delta_days_created, plot):
    plot.hist(x=[d / 365.0 for d in all_delta_days_created])[2][0]
    return


@app.cell
def _(all_delta_days_created, plot):
    plot.ecdf(x=[d / 365.0 for d in all_delta_days_created])
    return


@app.cell
def _(all_delta_days_created, np):
    int(np.median(all_delta_days_created))
    return


@app.cell
def _(all_delta_days_created, np):
    int(np.mean(all_delta_days_created))
    return


@app.cell
def _(mo):
    mo.md(r"""## Was there a newer version available at time of use? """)
    return


@app.cell
def _(Counter, codelist_jobs, codelists, datetime, sucessful_job_dates):
    _newer_available = []
    for _codelist_slug, _shas in codelist_jobs.items():
        for _sha in _shas:
            _run_date = sucessful_job_dates[_sha]
            for _codelist in codelists:
                if _codelist_slug and _codelist_slug.strip("/").startswith(
                    _codelist["slug"]
                ):
                    _versions_available_at_job_run = [
                        _v
                        for _v in _codelist["versions"]
                        if datetime.fromisoformat(_v["created_at"]).replace(tzinfo=None)
                        < _run_date
                    ]
                    _newer_codelists = True
                    for _version in sorted(
                        _versions_available_at_job_run, key=lambda x: x["created_at"]
                    ):
                        if _codelist_slug.strip("/") in [
                            _version["slug"],
                            _codelist["slug"] + "/" + _version["hash"],
                        ]:
                            _newer_codelists = False
                    _newer_available.append(_newer_codelists)
    Counter(_newer_available)

    return


if __name__ == "__main__":
    app.run()
