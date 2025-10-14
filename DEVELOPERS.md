# Notes for developers

## System requirements

### just

Follow installation instructions from the [Just Programmer's Manual](https://just.systems/man/en/chapter_4.html) for your OS.

Add completion for your shell. E.g. for bash:
```
source <(just --completions bash)
```

Show all available commands
```
just #  shortcut for just --list
```

### Environment Variables

Fetching variable definitions from the GitHub API requires a GitHub Personal Access Token (PAT) with `read` access permissions for repositories in the `opensafely` organisation.
This PAT must be set to the `GITHUB_TOKEN` environment variable, and can be added to a `.env` file which will be loaded automatically by `just` and has been included in `.gitignore`.


## Local development environment


Set up a local development environment with:
```
just devenv
```

## Running the project locally
```
just run
```
