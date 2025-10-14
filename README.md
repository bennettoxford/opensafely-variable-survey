# OpenSAFELY Variable Survey

This repository contains work by @bennettoxford/team-rsi to survey how "variables" are used by researchers in OpenSAFELY study.

A "variable" is a demographic or clinical feature of interest within an Electronic Health Record (EHR) that is used within an EHR study.

The work in this repository encompasses studies written using the deprecated [cohort-extractor](https://github.com/opensafely-core/cohort-extractor/) framework,
and those using its replacement - [ehrQL](https://github.com/opensafely-core/ehrql).

## cohort-extractor studies

The main entry point for cohort-extractor related code in this repository is `main.py` which is most conveniently run using the `just` command, `just run`.


`just run fetch` will fetch all variables from all studies in all repositories in the [`opensafely`](https://github.com/opensafely) organisation.
This requires a  suitable GitHub Personal Access Token (PAT),
more information on this is available in the [Developer Documentation](#developer-docs).

`just run notebooks` will start a local Marimo notebook server,
from which the `names.py` and `definitions.py` notebooks can be accessed.
These notebooks contain the analysis of variable names and definitions performed as part of this work,
and can be used a starting point for any desired future analysis.


## Developer docs

Please see the [additional information](DEVELOPERS.md).
