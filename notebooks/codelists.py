import marimo


__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _():
    import sys
    from collections import defaultdict

    import marimo as mo
    import pandas as pd
    import seaborn as sns

    return defaultdict, mo, pd, sns, sys


@app.cell
def _(mo):
    def data(path):
        return str(mo.notebook_location() / "public" / path)

    return (data,)


@app.cell
def _(data, pd):
    # load into dataframes
    df_codelists = pd.read_json(data("rsi-codelists-analysis.json"))
    df_codelist_versions = pd.DataFrame()
    for _, row in df_codelists.iterrows():
        codelist_slug = row["slug"]
        df_codelist_versions = pd.concat(
            [
                df_codelist_versions,
                pd.DataFrame(
                    [
                        {"codelist_slug": codelist_slug} | version
                        for version in row["versions"]
                    ]
                ),
            ]
        )

    # normalise to hash-based slugs for versions
    # some variables reference the same codelist version by hash or tag
    df_codelist_versions["normalised_slug"] = (
        df_codelist_versions["codelist_slug"] + "/" + df_codelist_versions["hash"]
    )
    normalised_version_slugs = {
        _row["slug"]: _row["normalised_slug"]
        for _, _row in df_codelist_versions.iterrows()
    } | {
        _row["normalised_slug"]: _row["normalised_slug"]
        for _, _row in df_codelist_versions.iterrows()
    }

    df_codelist_versions["slug"] = df_codelist_versions["normalised_slug"]
    df_codelist_versions.drop(columns="normalised_slug")

    df_codelists.set_index("slug", inplace=True)
    df_codelist_versions.set_index("slug", inplace=True)
    return df_codelist_versions, df_codelists, normalised_version_slugs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Overall counts from OpenCodelists
    ### Codelists with at least one under review/published version:
    """
    )
    return


@app.cell
def _(df_codelists):
    len(df_codelists)
    return


@app.cell
def _(df_codelists):
    df_codelists.groupby("coding_system").count()["name"].sort_values()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Under review/published versions""")
    return


@app.cell
def _(df_codelist_versions):
    len(df_codelist_versions)
    return


@app.cell
def _(df_codelist_versions, df_codelists):
    df_codelist_versions.merge(
        df_codelists, left_on="codelist_slug", right_index=True
    ).groupby("coding_system").count()["name"].sort_values()
    return


@app.cell
def _(data, pd):
    df_job = pd.read_csv(
        data("All.jobs-data-2025-11-10.18_25_26.csv"), index_col="sha"
    ).rename(columns={"_status": "status"})
    return (df_job,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Job data""")
    return


@app.cell
def _(df_job):
    len(df_job)
    return


@app.cell
def _(df_job):
    df_job.groupby("status").count()["url"].sort_values()
    return


@app.cell
def _(data, defaultdict, pd):
    ehrql_codelists = pd.read_json(data("ehrql_codelists.json"))
    signatures_to_shas = defaultdict(set[str])
    signatures_to_projects = dict()
    for _project, _row in ehrql_codelists.iterrows():
        if not isinstance(_row["projects"], dict):
            continue
        for _sha, _signature in _row["projects"].items():
            signatures_to_shas[_signature].add(_sha)
            signatures_to_projects[_signature] = _project
    df_repo_signatures = pd.DataFrame(
        [{"repo": v, "signature": k} for k, v in signatures_to_projects.items()]
    )
    return (
        df_repo_signatures,
        ehrql_codelists,
        signatures_to_projects,
        signatures_to_shas,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Repo signatures
    Unique combinations of variable definitions in a repo
    """
    )
    return


@app.cell
def _(df_repo_signatures):
    len(df_repo_signatures)
    return


@app.cell
def _(df_repo_signatures):
    # signatures per repo. "Count" is number of repos
    df_repo_signatures.groupby("repo").count().describe()
    return


@app.cell
def _(ehrql_codelists):
    ehrql_codelists
    return


@app.cell
def _(
    df_job,
    ehrql_codelists,
    normalised_version_slugs,
    pd,
    signatures_to_projects,
    signatures_to_shas,
    sys,
):
    # link ehrql codelist data and job server data
    df_variables = pd.DataFrame()  # id, sig, file, name
    df_jobs_variables = pd.DataFrame()  # sha, var_id
    df_variables_codelists = pd.DataFrame()  # var_id, codelist_version_slug
    iter_variable_id = iter(range(1, sys.maxsize))
    for _signature, _row in ehrql_codelists.iterrows():
        if not isinstance(_row["signatures"], dict):
            continue
        _project = signatures_to_projects[_signature]
        job_shas = {
            sha for sha in signatures_to_shas[_signature] if sha in df_job.index
        }
        for file, variables in _row["signatures"].items():
            for name, references in variables.items():
                variable_id = next(iter_variable_id)
                df_variables = pd.concat(
                    [
                        df_variables,
                        pd.DataFrame(
                            [
                                {
                                    "variable_id": variable_id,
                                    "signature": _signature,
                                    "file": file,
                                    "name": name,
                                }
                            ]
                        ),
                    ]
                )
                df_jobs_variables = pd.concat(
                    [
                        df_jobs_variables,
                        pd.DataFrame(
                            [
                                {"sha": sha, "variable_id": variable_id}
                                for sha in job_shas
                            ]
                        ),
                    ]
                )
                for _definition in references:
                    if len(_definition) in [2, 3]:
                        if not _definition[0]:
                            continue
                        _codelist_version_slug = normalised_version_slugs.get(
                            _definition[0].strip("/")
                        )
                        if not _codelist_version_slug:
                            continue
                        df_variables_codelists = pd.concat(
                            [
                                df_variables_codelists,
                                pd.DataFrame(
                                    [
                                        {
                                            "variable_id": variable_id,
                                            "codelist_version_slug": _codelist_version_slug,
                                        }
                                    ]
                                ),
                            ]
                        )
    df_variables.set_index("variable_id", inplace=True)
    return df_jobs_variables, df_variables, df_variables_codelists


@app.cell
def _(df_repo_signatures, df_variables, df_variables_codelists):
    # Count of projects using codelistversion
    df_repo_signatures.merge(df_variables.reset_index(), on="signature").merge(
        df_variables_codelists, on="variable_id"
    ).groupby("codelist_version_slug")["repo"].nunique().sort_values(ascending=False)
    return


@app.cell
def _(
    df_codelist_versions,
    df_repo_signatures,
    df_variables,
    df_variables_codelists,
):
    # Count of projects using codelist
    df_repo_signatures.merge(df_variables.reset_index(), on="signature").merge(
        df_variables_codelists, on="variable_id"
    ).merge(
        df_codelist_versions, left_on="codelist_version_slug", right_index=True
    ).groupby("codelist_slug")["repo"].nunique().sort_values(ascending=False)
    return


@app.cell
def _(df_variables, df_variables_codelists):
    # signatures using codelistversion
    df_variables.merge(
        df_variables_codelists, left_index=True, right_on="variable_id"
    ).groupby("codelist_version_slug").count()["signature"].sort_values(ascending=False)
    return


@app.cell
def _(df_codelist_versions, df_variables, df_variables_codelists):
    # signatures using codelistversion
    df_variables.merge(
        df_variables_codelists, left_index=True, right_on="variable_id"
    ).merge(
        df_codelist_versions, left_on="codelist_version_slug", right_index=True
    ).groupby("codelist_slug").count()["signature"].sort_values(ascending=False)
    return


@app.cell
def _(df_codelist_versions, df_variables, df_variables_codelists):
    # Number of versions of a given codelist in use by a variable
    # signatures using codelistversion
    df_variables.merge(
        df_variables_codelists, left_index=True, right_on="variable_id"
    ).merge(
        df_codelist_versions, left_on="codelist_version_slug", right_index=True
    ).groupby("codelist_slug")["codelist_version_slug"].nunique().sort_values(
        ascending=False
    )
    return


@app.cell
def _(df_job, df_jobs_variables, df_variables_codelists):
    # jobs using codelistversion
    df_job[df_job.status == "succeeded"].merge(df_jobs_variables, on="sha").merge(
        df_variables_codelists, on="variable_id"
    ).groupby("codelist_version_slug")["sha"].nunique().sort_values(ascending=False)
    return


@app.cell
def _(df_codelist_versions, df_job, df_jobs_variables, df_variables_codelists):
    # jobs using codelistversion
    df_job[df_job.status == "succeeded"].merge(df_jobs_variables, on="sha").merge(
        df_variables_codelists, on="variable_id"
    ).merge(
        df_codelist_versions, left_on="codelist_version_slug", right_index=True
    ).groupby("codelist_slug")["sha"].nunique().sort_values(ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Never-used codelist versions (ehrQL)""")
    return


@app.cell
def _(df_codelist_versions, df_variables_codelists):
    df_codelist_version_never_used = df_codelist_versions.merge(
        df_variables_codelists,
        right_on="codelist_version_slug",
        left_index=True,
        how="outer",
        indicator=True,
    )
    df_codelist_version_never_used = df_codelist_version_never_used[
        df_codelist_version_never_used._merge == "left_only"
    ][["codelist_version_slug"]].set_index("codelist_version_slug")
    df_codelist_version_never_used
    return (df_codelist_version_never_used,)


@app.cell
def _(df_codelist_version_never_used, df_codelist_versions, df_codelists):
    df_codelist_version_never_used.merge(
        df_codelist_versions, right_index=True, left_index=True
    ).merge(df_codelists, left_on="codelist_slug", right_index=True).groupby(
        "coding_system"
    ).count()["codelist_slug"].sort_values(ascending=False)
    return


@app.cell
def _(df_codelist_version_never_used, df_codelist_versions, df_codelists):
    df_codelist_version_never_used.merge(
        df_codelist_versions, right_index=True, left_index=True
    ).merge(df_codelists, left_on="codelist_slug", right_index=True).groupby(
        "owner"
    ).count()["codelist_slug"].sort_values(ascending=False)
    return


@app.cell
def _(mo):
    mo.md(r"""## Never-used codelist versions (ehrQL or cohort-extractor)""")
    return


@app.cell
def _(data, df_codelist_version_never_used, normalised_version_slugs, pd):
    # remove cohortextractor-used codelists
    cohort_extractor_codelist_versions = pd.read_json(
        data("cohort_extractor_codelists.json"), typ="series"
    )
    all_cohort_extractor_codelist_versions = {
        c for cs in cohort_extractor_codelist_versions.values for c in cs
    }
    all_cohort_extractor_codelist_versions = {
        _normalised_slug
        for slug in all_cohort_extractor_codelist_versions
        if (_normalised_slug := normalised_version_slugs.get(slug.strip("/")))
    }
    df_codelist_version_never_used_cohortextractor = df_codelist_version_never_used[
        ~df_codelist_version_never_used.index.isin(
            all_cohort_extractor_codelist_versions
        )
    ]
    df_codelist_version_never_used_cohortextractor
    return (df_codelist_version_never_used_cohortextractor,)


@app.cell
def _(
    df_codelist_version_never_used_cohortextractor,
    df_codelist_versions,
    df_codelists,
):
    df_codelist_version_never_used_cohortextractor.merge(
        df_codelist_versions, right_index=True, left_index=True
    ).merge(df_codelists, left_on="codelist_slug", right_index=True).groupby(
        "coding_system"
    ).count()["codelist_slug"].sort_values(ascending=False)
    return


@app.cell
def _(
    df_codelist_version_never_used_cohortextractor,
    df_codelist_versions,
    df_codelists,
):
    df_codelist_version_never_used_cohortextractor.merge(
        df_codelist_versions, right_index=True, left_index=True
    ).merge(df_codelists, left_on="codelist_slug", right_index=True).groupby(
        "owner"
    ).count()["codelist_slug"].sort_values(ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## How out of date are codelists when used?""")
    return


@app.cell
def _(
    df_codelist_versions,
    df_codelists,
    df_job,
    df_jobs_variables,
    df_variables_codelists,
    pd,
):
    # calculate time delta between codelist version creation and job execution
    df_timedelta = (
        df_codelist_versions.merge(
            df_variables_codelists, left_index=True, right_on="codelist_version_slug"
        )
        .merge(df_jobs_variables, on="variable_id")
        .merge(
            df_job[df_job.status == "succeeded"],
            on="sha",
            suffixes=["_codelist_version", "_job"],
        )
        .merge(
            df_codelists,
            left_on="codelist_slug",
            right_index=True,
            suffixes=["", "_codelist"],
        )
    )
    df_timedelta["timedelta"] = pd.to_datetime(
        df_timedelta.created_at_job, format="ISO8601", utc=True
    ) - pd.to_datetime(
        df_timedelta.created_at_codelist_version, format="ISO8601", utc=True
    )
    df_timedelta["deltadays"] = df_timedelta["timedelta"].apply(lambda x: x.days)
    return (df_timedelta,)


@app.cell
def _(df_timedelta):
    df_timedelta["deltadays"].describe()
    return


@app.cell
def _(df_timedelta):
    df_timedelta["deltadays"].hist()
    return


@app.cell
def _(df_timedelta):
    df_timedelta["deltadays"].apply(lambda x: float(x) / 365).hist()
    return


@app.cell
def _(df_timedelta, sns):
    sns.kdeplot(data=df_timedelta, x="deltadays", hue="coding_system")
    return


@app.cell
def _(df_timedelta, sns):
    sns.histplot(
        data=df_timedelta,
        x="deltadays",
        hue="coding_system",
        binwidth=100,
        y="coding_system",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""_N.B: No BNF since BNF codelists can't be used in OpenSAFELY_""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Was there a newer version available at time of use?""")
    return


@app.cell
def _(df_codelist_versions, df_timedelta):
    df_newer_available = df_timedelta[
        [
            "codelist_version_slug",
            "codelist_slug",
            "created_at_codelist_version",
            "created_at_job",
            "status_codelist_version",
            "variable_id",
            "sha",
        ]
    ]
    df_newer_available = df_newer_available.merge(
        df_codelist_versions.reset_index(), on="codelist_slug", suffixes=["", "_newer"]
    )
    df_newer_available = df_newer_available[
        (df_newer_available.codelist_version_slug != df_newer_available.slug)
        & (
            df_newer_available.created_at_codelist_version
            < df_newer_available.created_at
        )
    ]
    df_newer_available["newer_available"] = (
        df_newer_available.created_at_job > df_newer_available.created_at
    )
    return (df_newer_available,)


@app.cell
def _(df_newer_available):
    df_newer_available["job_variable_id"] = (
        df_newer_available["sha"] + "/" + df_newer_available["variable_id"].astype(str)
    )
    return


@app.cell
def _(df_newer_available):
    df_newer_available.groupby(["newer_available"])["job_variable_id"].nunique()
    return


@app.cell
def _(df_newer_available):
    df_newer_available[df_newer_available.newer_available].groupby(["status"])[
        "job_variable_id"
    ].nunique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## How often are "out of date"* codelist versions used?

    _ * if a new version of the codelist had been created at job urn time would there have been new codes? _

    N.B: A bit of uncertainty in these results, uses `compatible_release` feature of OpenCodelists which was introduced at end of 2022.
    """
    )
    return


@app.cell
def _(data, normalised_version_slugs, pd):
    coding_system_release_dates = pd.read_json(
        data("coding_system_release_dates.json"), typ="series"
    )
    version_compatible_release_dates = {
        _normalised_slug: v
        for k, v in pd.read_json(data("version_compat.json"), typ="series").items()
        if (_normalised_slug := normalised_version_slugs.get(k))
    }
    return coding_system_release_dates, version_compatible_release_dates


@app.cell
def _(version_compatible_release_dates):
    len(version_compatible_release_dates)
    return


@app.cell
def _(
    coding_system_release_dates,
    df_codelists,
    pd,
    version_compatible_release_dates,
):
    # find the valid_from date of the *next* coding system release after the latest compatible one
    version_out_of_date_at = []
    for (
        _codelist_version_slug,
        compatibility_dates,
    ) in version_compatible_release_dates.items():
        last_compatible_date = max(compatibility_dates)
        for _slug, coding_system in df_codelists["coding_system"].items():
            if _codelist_version_slug.startswith(_slug):
                break
        best_before_date = min(
            [
                _d
                for _d in coding_system_release_dates[coding_system]
                if _d > last_compatible_date
            ]
            or [None]
        )
        version_out_of_date_at.append(
            {
                "codelist_version_slug": _codelist_version_slug,
                "best_before_date": best_before_date,
            }
        )
    df_best_before_date = pd.DataFrame(version_out_of_date_at).set_index(
        "codelist_version_slug"
    )  # None == we're already at the latest version  # type: ignore
    return (df_best_before_date,)


@app.cell
def _(df_best_before_date):
    # No best before - therefore fully up to date at time of extract
    df_best_before_date.best_before_date.isnull().value_counts()
    return


@app.cell
def _(
    df_best_before_date,
    df_codelist_versions,
    df_codelists,
    df_job,
    df_jobs_variables,
    df_variables_codelists,
    pd,
):
    # codelists used after expiry
    df_used_after_expiry = (
        df_job[df_job.status == "succeeded"]
        .merge(df_jobs_variables, left_index=True, right_on="sha")
        .merge(df_variables_codelists, on="variable_id")
        .merge(
            df_codelist_versions,
            left_on="codelist_version_slug",
            right_index=True,
            suffixes=["_job", "codelist_version"],
        )
        .merge(df_codelists, left_on="codelist_slug", right_index=True)
        .merge(df_best_before_date, left_on="codelist_version_slug", right_index=True)
    )
    df_used_after_expiry["used_after_expiry"] = pd.to_datetime(
        df_used_after_expiry.created_at_job, format="ISO8601", utc=True
    ) > pd.to_datetime(df_used_after_expiry.best_before_date, yearfirst=True, utc=True)
    return (df_used_after_expiry,)


@app.cell
def _(df_used_after_expiry):
    df_used_after_expiry.used_after_expiry.value_counts()
    return


@app.cell
def _(df_used_after_expiry):
    df_used_after_expiry[df_used_after_expiry.used_after_expiry]
    return


@app.cell
def _(df_used_after_expiry):
    df_used_after_expiry.groupby("coding_system").used_after_expiry.value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""## Can we use version compatibility dates to derive most recent "true" update date for a codelist version"""
    )
    return


@app.cell
def _(
    df_codelist_versions,
    df_codelists,
    df_job,
    df_jobs_variables,
    df_variables_codelists,
    pd,
    version_compatible_release_dates,
):
    def _derive_true_updated_at(row):
        compatibility_dates = version_compatible_release_dates.get(
            row["codelist_version_slug"]
        )
        if not compatibility_dates or compatibility_dates == ["1900-01-01"]:
            return row["updated_at"]
        return max([cd for cd in compatibility_dates if cd < row["created_at_job"]])

    df_true_updated_at = (
        df_job[df_job.status == "succeeded"]
        .merge(df_jobs_variables, left_index=True, right_on="sha")
        .merge(df_variables_codelists, on="variable_id")
        .merge(
            df_codelist_versions,
            left_on="codelist_version_slug",
            right_index=True,
            suffixes=["_job", "_codelist_version"],
        )
        .merge(
            df_codelists,
            left_on="codelist_slug",
            right_index=True,
            suffixes=["", "_codelist"],
        )
    )

    df_true_updated_at["true_updated_at"] = df_true_updated_at.apply(
        lambda x: _derive_true_updated_at(x), axis=1
    )

    df_true_updated_at["timedelta"] = pd.to_datetime(
        df_true_updated_at.created_at_job, format="ISO8601", utc=True
    ) - pd.to_datetime(df_true_updated_at.true_updated_at, format="ISO8601", utc=True)
    df_true_updated_at["deltadays"] = df_true_updated_at["timedelta"].apply(
        lambda x: x.days
    )
    return (df_true_updated_at,)


@app.cell
def _(df_true_updated_at, sns):
    sns.histplot(df_true_updated_at["deltadays"], binwidth=100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Version compatibility dates are not reliable as of 2025/11/26 - bug found in their calculation which will have to be rectified to enable this analysis."""
    )
    return


if __name__ == "__main__":
    app.run()
