"""Compatibility CLI for the length-grouped SFT trainer."""

import sys
from trainers import sft_grouped as _impl

# Keep historical imports patchable: callers importing ``full_sft_packed`` get
# the implementation module, so monkeypatches affect the functions' globals.
sys.modules[__name__] = _impl
from trainers.sft_grouped import *
from trainers.sft_grouped import main


if __name__ == "__main__":
    main()
