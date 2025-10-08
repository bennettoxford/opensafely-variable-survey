import marimo


__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from itertools import chain, product

    import marimo as mo
    import pandas as pd
    from nltk.metrics.distance import jaro_winkler_similarity
    from scipy import stats

    return chain, jaro_winkler_similarity, json, mo, pd, product, stats


@app.cell
def _(json):
    # It'd be nice if this just worked. bleh
    # pd.read_json("cohort_extractor_variables.json")
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
    mo.md("""Number of studies per repo:""")
    return


@app.cell
def _(df):
    df.groupby("repo").study.nunique().describe()
    return


@app.cell
def _(mo):
    mo.md("""Number of variables per study:""")
    return


@app.cell
def _(df):
    df.groupby(["repo", "study"]).variable_name.nunique().describe()
    return


@app.cell
def _(df, mo):
    mo.md(
        f"""Number of unique variable names: {df.variable_name.str.lower().nunique()} / {len(df.index)}"""
    )
    return


@app.cell
def _(mo):
    mo.md("""## Most common variable names (exact matching):""")
    return


@app.cell
def _(df):
    df.variable_name.str.lower().value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fuzzy name clustering

    On the assumption that _similar_ names will be most similar near the beginning of the string (abbreviations etc.), try using the [Jaro-Winkler](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance) edit distance, which weights edits based on their position.

    We can try supervised or unsupervised clustering of names based on this distance metric, success may depend on the distribution of the pairwise distances.
    """
    )
    return


@app.cell
def _(df, jaro_winkler_similarity, product):
    from collections import defaultdict

    unique_variable_names = sorted(df.variable_name.str.lower().unique())
    similarities = defaultdict(list)
    for a, b in product(
        enumerate(unique_variable_names), enumerate(unique_variable_names)
    ):
        # don't compare against self, and don't compare sim("bar","foo") if we've computed sim("foo","bar")
        if b[0] >= a[0]:
            continue
        sim = jaro_winkler_similarity(a[1], b[1])
        similarities[a[0]].append(sim)
    return defaultdict, similarities, unique_variable_names


@app.cell
def _(chain, mo, similarities, stats):
    mo.md(
        f"""Anderson-Darling test for fit with normal distribution result was: '{stats.anderson(list(chain.from_iterable(similarities.values()))).fit_result.success}' therefore we can assume normally distributed."""
    )
    return


@app.cell
def _(chain, defaultdict, similarities, stats):
    zscores_flat = stats.zscore(
        list(chain.from_iterable(similarities.values())), axis=None
    )
    zscores = defaultdict(list)
    zs_ix = 0
    for sim_k, sim_v in similarities.items():
        for v in sim_v:
            zscores[sim_k].append(zscores_flat[zs_ix])
            zs_ix += 1
    return (zscores,)


@app.cell
def _(zscores):
    # cluster where zscore > 2
    # zs_gt_two = zscores > 2
    clusters = [set()]
    # for i,j in product(range(len(unique_variable_names)),range(len(unique_variable_names))):
    for i, v_zs in zscores.items():
        for j, zscore in enumerate(v_zs):
            if zscore <= 2:
                continue
            found = False
            for cluster in clusters:
                if {i, j}.intersection(cluster):
                    cluster |= {i, j}
                    found = True
            if not found:
                clusters.append({i, j})
    return (clusters,)


@app.cell
def _(clusters):
    set([len(cluster) for cluster in clusters])  # len(clusters[1])
    return


@app.cell
def _(clusters):
    [cluster for cluster in clusters if len(cluster) == 2]
    return


@app.cell
def _(jaro_winkler_similarity, unique_variable_names):
    jaro_winkler_similarity(*unique_variable_names[634:636])
    return


if __name__ == "__main__":
    app.run()
