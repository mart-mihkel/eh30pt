# Agents

eh30pt (prompt tuning on Est-Health 30) is a single-package Python project for prompt-tuning, fine-tuning, and few-shot evaluation of HuggingFace models on Estonian clinical NLP datasets. Given a model and a dataset, it trains or evaluates under an open MLflow run and registers the best checkpoint.

## Repository layout

```
src/eh30pt/           # the package, installed as `cli`
  cli.py                # Typer app, thin option definitions only
  constants.py          # shared paths (e.g. LOGDIR)
  logging.py            # rich-backed shared logger, tqdm-safe
  tracking.py           # MLflow run management, config-hash resume
  types.py              # shared Literals, TypedDicts, Protocols
  datasets/             # per-dataset pipelines + registry + truncation
  modeling/             # arch detection, loading, tokenization, collation, metrics, trainer, tuning
  scripts/              # command implementations: finetune, ptune, fewshot, export
test/                 # tests, lint-relaxed
  unit/                 # fast pure-logic tests, no models or network
  integration/          # dataset loading and forward passes on tiny-random models
  e2e/                  # cli tests
vendor/               # gguf conversion scripts from llama.cpp, lint-ignored
notebooks/            # exploratory marimo notebook analysis, lint-relaxed
justfile              # project commands
pyproject.toml        # project metadata, ruff and pytest config
prek.toml             # ruff-format, ruff-check, ty-check hooks
compose.yml           # postgres service for local development
sql/                  # initdb seed scripts for the dev database
```

## Commands

```bash
just sync                                   # uv sync virtualenv, install prek hooks
just check                                  # prek checks, all files by default
just test [flags]                           # pytest with flag/varitaic passthrough

uv run --no-sync pytest [--run-slow] [path]          # run pytest directly
uv run --no-sync ruff format                         # format Python
uv run --no-sync ruff check --output-format concise  # lint Python
uv run --no-sync ty check --output-format concise    # type-check Python
```

## Conventions

- There is a single uv package with one lockfile.
- Datasets are reached through a registry rather than imported directly.
- MLflow runs are reattached by config hash and the checkpoint directory is derived from the run id.
- The local `log/` copy of checkpoints is a fast cache, not an identity.
- Use the logger from the logging module instead of configuring a per-module logger.
- Shared types live in `types.py`.
- Python docstrings follow numpy convention and are verified by pydoclint.
- Tests are split between unit, integration and end to end.
- Pytest raises warnings as errors, so new warnings fail the suite rather than accumulating silently.
- The `notebooks` dependency group is dev-only and never imported outside of notebooks.

## Guidelines

- Do not ignore linter or typechecker errors without permission.
- Use type casting or asserts if needed instead of loosening a type.
- Always add precise and exhaustive type annotations to everything.
- Prefer exhaustive types over bare containers, a pandera-typed `DataFrame[Schema]` over a plain `DataFrame`. Use precise generics or designated typing classes over bare `list`/`tuple`/`dict`, don't be afraid of creating an entire `NamedTuple`, `TypedDict` or `Protocol` for small use cases. For bigger use cases use dataclasses or pydantic models.
- Avoid writing walls of text with rich markdown syntax when documenting code objects. Prefer using numpy docstrings with appropriate attribute, parameter, raise and return sections.
- All significant changes must be tested. Add or update focused tests for semantic changes when existing coverage does not already establish the intended behavior.
- Look to see if your tests could go in an existing file before adding a new file for your tests.
- Get your tests to pass. If you didn't run the tests, your code does not work.
- Follow existing code style. Check neighboring files for patterns.
- Run `just check` after making changes.
- Before writing significant amounts of new code, look for existing utilities or mechanisms that could solve the problem. Avoid expanding the task to unrelated issues, but do not confuse keeping the task focused with minimizing the size of the implementation. Prefer addressing the underlying architectural problem over adding a localized workaround, even when doing so requires a substantial refactor or rearchitecture. Ask the user for guidance if in doubt about whether to attempt a larger refactor or not.
- Try hard to avoid patterns that require `getattr` or `setattr`. Instead, try to encode those constraints in the type system. Don't be afraid to write code that's more verbose or requires largeish refactors if it enables you to avoid these unsafe calls.
- Don't use comments to narrate code, but do use them to explain invariants and why something unusual was done a particular way
- Make sure that a comment will make sense to somebody who's reading the code for the first time. Prefer plain language, avoid jargon, and don't be afraid to be more verbose if it's necessary to explain something well.
- Avoid writing walls of rich markdown text when documenting code objects. Use a docstring format.
