"""V3 pretraining entry point.

The implementation remains in :mod:`pretrain` for backwards compatibility;
this module provides the stable trainer boundary used by the CLI and future
iterations can move the implementation here without changing the command.
"""


def run_v3(parsed_args):
    # Import lazily to avoid a circular import while ``pretrain`` is loading.
    from pretrain import run_v3 as _run_v3

    return _run_v3(parsed_args)
