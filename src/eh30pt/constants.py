"""Global constants."""

import os
from pathlib import Path
from typing import Final

LOGDIR: Final = Path(os.environ.get("EH30PT_LOGDIR", "log"))
"""
Destination for logs and model checkpoints, overridable with EH30PT_LOGDIR.

Relative paths resolve against the working directory where training runs.
"""

SENTINEL_TOKEN: Final = -100
"""
Sentinel value used to mark positions that should be ignored by the loss/metrics.
"""

PAD_MULTIPLE: Final = 8
"""
GPUs run faster when tensor sizes line up with their memory/layout "tile" sizes.
"""

UNSET_MAX_LENGTH: Final = int(1e15)
"""
Tokenizers without a real length limit report a ~1e30 sentinel.
Treat anything at or above this threshold as no limit configured.
"""
