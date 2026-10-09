# Development

Use [just](https://just.systems/man/en/) as the command runner and [uv](https://docs.astral.sh/uv/getting-started/installation/) as the Python project manager.
Run just without arguments to list all available development commands.

## Setup

Set up the virtualenv and install the pre-commit hooks with:

```bash
just sync
```

The package installs a command line interface into the virtualenv.
Run it with `uv run ehpt --help` to see the available commands.

## Services

Local development uses postgres, llama and label-studio services.

```bash
just up   [--detach]
just down [--volumes]
```

## Checks

All code changes must pass the pre-commit checks and tests.
See the [tests](./tests.md) document for the test layout, fixtures, and options.

```bash
just prek  # pre-commit checks
just test  # fast unit and integration test suite
just e2e   # slow end to end test suite
```

## Notebooks

Exploratory analysis lives in the notebooks directory as marimo notebooks.

```bash
just marimo  # marimo server
```

## Remote Machines

The working tree can be uploaded to a remote host or a wheel to object storage.

```bash
just rsync [--host ...]  [--commit]      # upload working tree to remote host
just msync [--alias ...] [--prefix ...]  # upload wheel to minio object storage
```
