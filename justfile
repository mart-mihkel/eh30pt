set dotenv-load

[doc('list recipes')]
[group('meta')]
[private]
@info:
    just --list

[doc('sync virtualenv')]
[group('build')]
@sync:
    uv sync --compile-bytecode
    uv run --no-sync prek install

[arg('host', long, short, help='remote host and target dir')]
[arg('commit', long, short, pattern='true|false', value='true', help='no dry run')]
[doc('upload to remote host')]
[group('build')]
rsync commit='false' host=env('REMOTE'):
    rsync --verbose --archive --delete \
        {{ if commit == 'true' { '' } else { '--dry-run' } }} \
        --exclude-from .gitignore \
        --exclude .pytest_cache \
        --exclude .ruff_cache \
        --exclude .git \
        . {{ host }}

[doc('build wheel')]
[group('build')]
wheel:
    uv build --wheel

[doc('serve marimo')]
[group('dev')]
marimo *args='notebooks':
    uv run --no-sync marimo edit {{ args }}

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

[doc('run pre-commit checks')]
[arg('staged', long, short, pattern='true|false', value='true', help='only check staged files')]
[group('test')]
@check staged='false':
    prek run {{ if staged == 'true' { '' } else { '--all-files' } }} 

[doc('run test suite')]
[group('test')]
@test *args='--quiet':
    uv run --no-sync pytest {{ args }}
