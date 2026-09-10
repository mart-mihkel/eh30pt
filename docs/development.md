# Development

Use [just](https://just.systems/man/en/) as the command runner and [uv](https://docs.astral.sh/uv/getting-started/installation/) as the Python project manager.
Run just without arguments to list all available development commands.

## Setup

Set up the virtualenv and install the pre-commit hooks with:

```bash
just sync
```

The package installs a command line interface into the virtualenv.
Run it with `uv run cli --help` to see the available commands.

## Services

Local development uses a Postgres database running as a Docker service.

```bash
just up [--detach]
just down [--volumes]
```

## Checks

All code changes must pass the pre-commit checks and tests.
See the [tests](./tests.md) document for the test layout, fixtures, and options.

```bash
just check  # pre-commit checks such as formatter, linter and typechecker
just test   # test suite
```

## Notebooks

Exploratory analysis lives in the notebooks directory as marimo notebooks.

```bash
just marimo  # run marimo server
```

## Remote Machines

The working tree can be uploaded to a remote host or the app can be packaged into a wheel.

```bash
just rsync [--host ...] [--commit]  # upload working tree to a remote host   (HPC)
just wheel                          # package into a wheel for manual upload (SAPU)
```
