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

[arg('host', long, short, help='remote host and target dir')]
[arg('commit', long, short, pattern='true|false', value='true', help='no dry run')]
[doc('upload working tree to remote host')]
[group('build')]
rsync commit='false' host=env('REMOTE'):
    rsync --verbose --archive --delete \
        {{ if commit == 'true' { '' } else { '--dry-run' } }} \
        --exclude-from .gitignore \
        --exclude .pytest_cache \
        --exclude .ruff_cache \
        --exclude .git \
        . {{ host }}

[doc('run pre-commit checks')]
[group('lint')]
@check:
    uv run --no-sync ruff format
    uv run --no-sync ruff check --fix
    uv run --no-sync ty check --fix

[doc('run test suite')]
[group('test')]
@test:
    uv run --no-sync pytest --quiet 
