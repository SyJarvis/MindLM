"""Compatibility CLI for the legacy fixed-length SFT trainer."""

import sys
from trainers import sft_legacy as _impl

# Preserve the historical module surface for imports and monkeypatching.
sys.modules[__name__] = _impl
from trainers.sft_legacy import *
from trainers.sft_legacy import main


if __name__ == "__main__":
    main()
