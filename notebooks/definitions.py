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
def _(mo):
    mo.md(
        r"""
    # Analysis of variation in cohort extractor variable definitions

    ## Summary statistics
    """
    )
    return


@app.cell
def _(df, mo):
    mo.md(
        f"""Number of exact unique variable definitions: {df.variable_definition.str.lower().nunique()} / {len(df.index)}"""
    )
    return


@app.cell
def _(df, mo):
    mo.md(
        f"""Number of whitespace-removed unique variable definitions: {df.variable_definition.str.lower().str.replace(pat=r"\s+", repl="", regex=True).nunique()} / {len(df.index)}"""
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
    df_most_definitions = (
        df[df.variable_name != "index_date"]
        .groupby("variable_name")
        .variable_definition.nunique()
        .rename("unique_definitions")
        .sort_values(ascending=False)
    )

    df_most_definitions
    return (df_most_definitions,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Individual definition analysis

    ### Age:
    """
    )
    return


@app.cell
def _(df):
    df[df.variable_name == "age"].variable_definition.value_counts()
    return


@app.cell
def _(mo):
    mo.md(
        r"""Per-variable expectations definitions (a feature of the cohort extractor API) are a source of spurious variation, let's remove them in the next tables."""
    )
    return


@app.cell
def _():
    import ast

    def remove_expectations(variable: str) -> str:
        mod = ast.parse(variable)
        for b in mod.body:
            if not isinstance(b, ast.Expr):
                continue
            if isinstance(b.value, ast.Call):
                for _i, kw in enumerate(b.value.keywords):
                    if kw.arg == "return_expectations":
                        mod.body[0].value.keywords.pop(_i)
        return ast.unparse(mod)

    return (remove_expectations,)


@app.cell
def _(df_most_definitions, mo):
    var_ddl = mo.ui.dropdown(options=df_most_definitions.head(10).index)
    var_ddl
    return (var_ddl,)


@app.cell
def _(mo, var_ddl):
    selected_var = var_ddl.value
    mo.md(f"### {selected_var.title() if selected_var else 'No variable selected'}:")
    return


@app.cell
def _(df, remove_expectations, var_ddl):
    df[df.variable_name == var_ddl.selected_key].variable_definition.apply(
        remove_expectations
    ).value_counts()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
