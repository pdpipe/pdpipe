# This file is part of pdpipe.
# https://github.com/pdpipe/pdpipe

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2024, Jirka Borovec <***@gmail.com>

import os

_PATH_HERE = os.path.dirname(__file__)
_PATH_VERSION = os.path.join(_PATH_HERE, "version.info")
_RELEASING_PROCESS = os.getenv("RELEASING_PROCESS", "0") == "1"

with open(_PATH_VERSION) as fopen:
    __version__ = fopen.read().strip()


def _get_git_sha() -> str:
    from subprocess import DEVNULL, check_output

    args = ["git", "rev-parse", "--short", "HEAD"]
    out = check_output(args, stderr=DEVNULL)  # noqa: S603
    return out.decode("utf-8").strip()


if not _RELEASING_PROCESS:
    try:
        sha_short = _get_git_sha()
        # print(f"Version enriched with git commit hash: {__version__}.")
    except Exception:
        # print("Failed to get the git commit hash,"
        #       f" falling back to base version {__version__}.")
        sha_short = ""
    __version__ += f".dev+{sha_short}" if sha_short else ".dev"


__all__ = ["__version__"]
