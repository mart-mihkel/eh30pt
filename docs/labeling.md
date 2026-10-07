# Labeling

`eh30pt create-task` reads a run's prediction shards, builds the tasks and the
labeling config from them, then creates a Label Studio project through the API
and imports the tasks into it. Annotators open the project and work from the
model's highlights rather than an empty span.

## Setup

Label Studio needs a personal access token, created under
*Account & Settings -> Personal Access Tokens* after enabling them in
*Organization -> Settings -> Access Token Settings*. Put it in `.env`:

```
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_URL is optional, it defaults to localhost:8080
LABEL_STUDIO_TOKEN=<personal access token>
```

`.env` is gitignored. The URL also has a `--label-studio-url` flag and the
`LABEL_STUDIO_URL` environment variable.

## Usage

```bash
eh30pt create-task --mlflow-run-id <run> [--dataset D] [--title T]
```

`--dataset` defaults to whatever the run was tracked with, read back from its
`config.json` (falling back to run params). Pass it explicitly only for runs
that record no dataset.

`--title` defaults to `eh30pt-<dataset>`. Each call creates a new project, so
two runs of the same dataset never share an annotation set. The command prints
the project id and the URL to open.

## Label interface

`span_config` renders three choice boxes side by side in a flex row:

- `discont` — Drug discontinuation
- `other` — Not discontinuation
- `meta` — Meta

Two things to know when editing `extract.py`. Boxes must never share a label
string: Label Studio exports only the chosen string, so `"Other"` in two boxes
collapses into one label on export, which is why the second box offers
`"Other event"`. And the header highlight uses `:has(input:checked)` to
recolour a border that is always drawn, because anything appearing or
disappearing on click reflows the page and made the boxes jump around.

## Tests

```bash
just test test/unit/test_labeling_studio.py          # response parsing
just test test/integration/test_streaming.py         # task and config building
just e2e test/e2e/test_labeling.py                   # against a live server
```

The end to end tests skip unless `LABEL_STUDIO_TOKEN` is set, and they create
and delete their own projects.

## Label Studio resources

- The server defaults to `sqlite` at `test.db` in the working directory, which
  is gitignored. Start it elsewhere if you would rather keep the repo clean.
- A personal access token is a JWT refresh token, so plain HTTP calls must
  exchange it at `POST /api/token/refresh` for a short-lived access token and
  send that as `Authorization: Bearer`. `StudioClient` does the exchange once
  and caches the result.
- Tasks accept no `predictions` on creation, but the bulk import endpoint does,
  which is what keeps pushing to one request per 100 tasks.
- The task list endpoint is paginated with `page`/`page_size` and returns items
  under `tasks`; the rest of the API uses `next`/`results`.
- A delete answers 204 with no body.