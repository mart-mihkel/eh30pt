# Tests

## Layout

```
test/
  e2e/          # end to end calls to the cli
  integration/  # integration tests
  unit/         # quick pure logic tests
```

## Conventions

Do not create mocks or fake data generation workflows in unit tests.
Instead use fixtures from integration tests or create ones with real underlying objects.
Keep mocking and faking to absolute minimum.

## Running

End to end training runs are marked slow and are skipped by default.
Pass the run-slow flag to include them.

```bash
pytest [--cov] [--run-slow] [--numprocesses n] [--record-mode rewrite] [path]
just test [[args]]
```

Some tests talk to a local llama server, the responses are recorded and resused by default.
To re-record them, use record-mode rewrite, which requires a running llama server on localhost port 8080.
Database integration tests provision their own seeded postgres via testcontainers.
