"""Global constants."""

from pathlib import Path
from typing import Final

LOGDIR: Final = Path("log")
"""
Destination for logs and model checkpoints.
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
