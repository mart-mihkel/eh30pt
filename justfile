set dotenv-load

[doc('list recipes')]
[group('meta')]
[private]
@info:
    just --list --unsorted

[doc('sync virtualenv')]
[group('build')]
@sync:
    uv sync --compile-bytecode
    uv run --no-sync prek install

[arg('host', long, help='remote host and target dir')]
[arg('commit', long, pattern='true|false', value='true', help='no dry run')]
[doc('upload tree to remote host')]
[group('build')]
rsync commit='false' host=env('RSYNC_HOST'):
    rsync --verbose --archive --delete \
        {{ if commit == 'true' { '' } else { '--dry-run' } }} \
        --exclude-from .gitignore \
        --exclude .pytest_cache \
        --exclude .ruff_cache \
        --exclude .git \
        . {{ host }}

[arg('alias', long, help='minio client host alias')]
[arg('prefix', long, help='object storage target path')]
[doc('upload wheel to object storage')]
[group('build')]
msync alias=env('MSYNC_ALIAS') prefix=env('MSYNC_PREFIX'):
    @uv build --wheel --clear
    mc cp dist/*.whl "{{ alias }}/{{ prefix }}"

[doc('serve marimo')]
[group('dev')]
marimo *args='--watch':
    uv run --no-sync marimo edit {{ args }}

[doc('run pre-commit')]
[group('dev')]
@prek:
    uv run --no-sync ruff format
    uv run --no-sync ruff check --fix
    uv run --no-sync ty check --fix

[doc('run all tests')]
[group('dev')]
@test *args='--quiet -n 4':
    COLUMNS=80 uv run --no-sync pytest {{ args }}

[doc('run e2e tests')]
[group('dev')]
@e2e *args='--quiet -n 2':
    COLUMNS=80 uv run --no-sync pytest test/e2e --run-slow {{ args }}

[doc('build docker services')]
[group('docker')]
@build *args:
    docker compose build {{ args }}

[doc('start docker services')]
[group('docker')]
@up *args:
    docker compose up {{ args }}

[doc('stop compose services')]
[group('docker')]
@down *args:
    docker compose down {{ args }}
