# Agents

eh30pt (prompt tuning on Est-Health 30) is a single-package python project for fine-tuning, peft methods and few-shot evaluation of trasformer models on Estonian clinical data.

## Layout

```
deploy/               # local label-studio init scripts
docs/                 # documentation markdown files
notebooks/            # exploratory marimo notebooks
run/                  # experiment run scripts
sql/                  # initdb seed scripts for the dev database
src/eh30pt/           # the package, installed as `ehpt` command line app
  cli/                  # command line interface and script implementations
  datasets/             # per-dataset pipelines, registry and common utils
  esthealth/            # esthalth database orm schemas and queries
  labeling/             # label studio tasks, configs and api client
  modeling/             # arch detection, loading, tokenization, collation, metrics, training, tuning
  openai/               # OpenAI API wrangling helpers
  tracking/             # MLflow run management, config-hash resume
test/                 # tests, lint-relaxed
  unit/                 # pure-logic tests, no models, no network, no fakes, no mocks
  integration/          # integrations tests using real models and objects
  e2e/                  # tests which invoke the cli
tex/                  # article submitted to medical informatics europe
justfile              # development commands
pyproject.toml        # project metadata and tool configs
prek.toml             # pre-commit hooks
compose.yml           # postgres service for local development
```

## Commands

```bash
just sync         # uv sync virtualenv, install prek hooks
just prek         # pre-commit checks, all files by default
just test [args]  # pytest with varitaic arg passthrough
```

## Conventions

- There is a single uv package with one lockfile.
- Datasets are reached through a registry rather than imported directly.
- Use the logger from `logging.py` instead of configuring a per-module logger.
- Shared types live in `types.py`.
- Python docstrings are one line high level overviews with no markdown syntax.
- Use fluent `mlflow` syntax instead of a client.
- Tests are split between unit, integration and end to end.
- Pytest raises warnings as errors, so new warnings fail the suite rather than accumulating silently.
- No mocks or fake data in unit tests.
- Use tqdm for loading bars or when iterating data streams.
- The `notebooks` dependency group is dev-only and never imported outside of notebooks.
- The development database and seed scripts are all fake, the real database is massive, classified and unreachable to you.
- See [docs/](./docs/) for testing and development overviews.

## Guidelines

- **NEVER** run git commands.
- Ask for permission before ignoring linter or typechecker errors (`noqa`, `ty: ignore`).
- Use type casting or asserts if needed instead of loosening a type.
- Always add precise and exhaustive type annotations to everything.
- Prefer exhaustive types over bare containers, a pandera-typed `DataFrame[Schema]` over a plain `DataFrame`. Use precise generics or designated typing classes over bare `list`/`tuple`/`dict`, don't be afraid of creating an entire `NamedTuple`, `TypedDict` or `Protocol` for small use cases. For bigger use cases use dataclasses or pydantic models.
- Avoid writing walls of text with rich markdown syntax when documenting code objects.
- All significant changes must be tested. Add or update focused tests for semantic changes when existing coverage does not already establish the intended behavior.
- Look to see if your tests could go in an existing file before adding a new file for your tests.
- Get your tests to pass. If you didn't run the tests, your code does not work.
- Follow existing code style. Check neighboring files for patterns.
- Run `just prek` after making changes.
- Before writing significant amounts of new code, look for existing utilities or mechanisms that could solve the problem. Avoid expanding the task to unrelated issues, but do not confuse keeping the task focused with minimizing the size of the implementation. Prefer addressing the underlying architectural problem over adding a localized workaround, even when doing so requires a substantial refactor or rearchitecture. Ask the user for guidance if in doubt about whether to attempt a larger refactor or not.
- Try hard to avoid patterns that require `getattr` or `setattr`. Instead, try to encode those constraints in the type system. Don't be afraid to write code that's more verbose or requires largeish refactors if it enables you to avoid these unsafe calls.
- Don't use comments to narrate code, but do use them to explain invariants and why something unusual was done a particular way
- Make sure that a comment will make sense to somebody who's reading the code for the first time. Prefer plain language, avoid jargon, and don't be afraid to be more verbose if it's necessary to explain something well.
- Avoid writing walls of rich markdown text when documenting code objects. Use a docstring format.
- Try not to create module level constatns, inline them or create a function parameter with a default value.
