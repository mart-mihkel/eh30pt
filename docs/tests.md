# Tests

## Layout

```
test/
  e2e/          # end to end calls to the cli
  integration/  # integration tests
  unit/         # quick pure logic tests
  conftest.py   # configuration and fixtures
```

## Running

Training runs are marked slow and are skipped by default. Pass the run-slow flag to include them.

```bash
pytest [--run-slow] [--record-mode rewrite] [path]
just test [[args]]
```

Some tests talk to a local llama server, the responses are recorded and resused by default.
To re-record them, use record-mode rewrite, which requires a running llama server on localhost port 8080.
