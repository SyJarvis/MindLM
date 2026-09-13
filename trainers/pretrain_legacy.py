"""Legacy pretraining entry point.

The implementation remains in :mod:`pretrain` for backwards compatibility;
this module provides the stable trainer boundary used by the CLI.
"""


def run_legacy(parsed_args):
    # Import lazily to avoid a circular import while ``pretrain`` is loading.
    from pretrain import run_legacy as _run_legacy

    return _run_legacy(parsed_args)
