import marimo


__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    import marimo as mo
    import pandas as pd

    return json, mo, pd


@app.cell
def _():
    # It'd be nice if this just worked. bleh
    # pd.read_json("cohort_extractor_variables.json")
    return


@app.cell
def _(json):
    # let's wrangle it manually
    with open("cohort_extractor_variables.json") as f:
        variables = json.load(f)
    return (variables,)


@app.cell
def _(variables):
    flattened = []
    for repo, studies in variables.items():
        for study, study_variables in studies.items():
            for variable in study_variables:
                flattened.append(
                    {
                        "repo": repo,
                        "study": study,
                        "variable_name": variable[0],
                        "variable_definition": variable[1],
                        "variable_line_number": variable[2],
                    }
                )
    return (flattened,)


@app.cell
def _(flattened, pd):
    df = pd.DataFrame(flattened)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, mo):
    mo.md(text=f"Number of repos:{len(df.repo.unique())}")
    return


@app.cell
def _(mo):
    mo.md("Number of studies per repo:")
    return


@app.cell
def _(df):
    df.groupby("repo").study.nunique().describe()
    return


@app.cell
def _(mo):
    mo.md("Number of variables per study:")
    return


@app.cell
def _(df):
    df.groupby(["repo", "study"]).variable_name.nunique().describe()
    return


@app.cell
def _(df, mo):
    mo.md(
        f"Number of unique variable names: {df.variable_name.str.lower().nunique()} / {len(df.index)}"
    )
    return


if __name__ == "__main__":
    app.run()
