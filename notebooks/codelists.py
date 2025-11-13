import marimo


__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import csv
    import json
    from collections import Counter, defaultdict
    from pathlib import Path

    return Counter, Path, csv, defaultdict, json


@app.cell
def _(Path, csv):
    jobs = list(
        csv.DictReader(Path("data/All.jobs-data-2025-11-10.18_25_26.csv").open())
    )
    job_shas = {job["sha"] for job in jobs}
    successful_job_shas = {job["sha"] for job in jobs if job["_status"] == "succeeded"}
    return job_shas, successful_job_shas


@app.cell
def _(Path, json):
    codelists = json.load(Path("data/rsi-codelists-analysis.json").open())
    return (codelists,)


@app.cell
def _(codelists):
    codelists
    return


@app.cell
def _(codelists):
    # total number of codelists with at least one under review/published version
    len(codelists)
    return


@app.cell
def _(codelists):
    # total number of published/under review codelist versions
    sum([len(codelist["versions"]) for codelist in codelists])
    return


@app.cell
def _(Path, json):
    ehrql_codelists = json.load(Path("data/ehrql_codelists.json").open())
    return (ehrql_codelists,)


@app.cell
def _(defaultdict, ehrql_codelists):
    signatures_to_shas = defaultdict(set[str])
    signatures_to_projects = defaultdict(str)
    for _project, _hashes in ehrql_codelists["projects"].items():
        for _sha, _signature in _hashes.items():
            signatures_to_shas[_signature].add(_sha)
            signatures_to_projects[_signature] = _project

    return signatures_to_projects, signatures_to_shas


@app.cell
def _(
    defaultdict,
    ehrql_codelists,
    job_shas,
    signatures_to_projects,
    signatures_to_shas,
):
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


@app.function
def sorted_dict(dict_to_sort: dict, reverse=True) -> dict:
    return {
        _k: _v
        for _k, _v in sorted(dict_to_sort.items(), key=lambda x: x[1], reverse=reverse)
    }


@app.cell
def _(codelist_jobs):
    # codelist versions in any job
    sorted_dict({_k: len(_v) for _k, _v in codelist_jobs.items()})
    return


@app.cell
def _(codelist_jobs, successful_job_shas):
    # codelist versions in successful jobs
    sorted_dict(
        {
            _k: len(_v.intersection(successful_job_shas))
            for _k, _v in codelist_jobs.items()
        }
    )
    return


@app.cell
def _(codelist_variables):
    # numbers of variables featuring codelist version
    sorted_dict(codelist_variables)
    return


@app.cell
def _(codelist_projects):
    # numbers of projects featuring codelist version
    sorted_dict(
        {codelist: len(projects) for codelist, projects in codelist_projects.items()}
    )
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


if __name__ == "__main__":
    app.run()
