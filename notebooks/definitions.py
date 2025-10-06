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
def _(json, pd):
    with open("cohort_extractor_variables.json") as f:
        variables = json.load(f)
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
    df = pd.DataFrame(flattened)
    return (df,)


@app.cell
def _(df, mo):
    mo.md(
        f"Number of exact unique variable definitions: {df.variable_definition.str.lower().nunique()} / {len(df.index)}"
    )
    return


@app.cell
def _(df, mo):
    mo.md(
        f"Number of whitespace-removed unique variable definitions: {df.variable_definition.str.lower().str.replace(pat=r'\s+', repl='', regex=True).nunique()} / {len(df.index)}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Most common variable definitions (excluding index date):""")
    return


@app.cell
def _(df):
    df[df.variable_name != "index_date"].variable_definition.str.lower().str.replace(
        pat=r"\s+", repl="", regex=True
    ).value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Variation in definition within variables of same name""")
    return


@app.cell
def _(df):
    df[df.variable_name != "index_date"].groupby(
        "variable_name"
    ).variable_definition.nunique().sort_values(ascending=False)
    return


@app.cell
def _(df):
    df[df.variable_name == "age"].variable_definition.value_counts()
    return


@app.cell
def _(df):
    df[df.variable_name == "ethnicity"].variable_definition.value_counts()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
