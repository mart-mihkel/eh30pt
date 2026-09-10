"""
MLflow experiment tracking.

Runs are the source of truth for resume: a run is reattached by its config
hash, and its checkpoint directory is derived from the run id. The local
``log/`` copy of checkpoints is a fast cache, not an identity.
"""

from eh30pt.tracking.collect import collect_runs
from eh30pt.tracking.context import RunContext
from eh30pt.tracking.resume import ConfigMismatchError, build_run_name, open_run

__all__ = [
    "ConfigMismatchError",
    "RunContext",
    "build_run_name",
    "collect_runs",
    "open_run",
]
