"""Configuring pytest for pdpipe tests."""

import os
import sys
from pathlib import Path

import xdg
import pytest


# make
sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))


@pytest.fixture(scope='session', autouse=False)
def pdpipe_tests_dir_path() -> str:
    cache_dpath: Path
    cache_dpath = xdg.xdg_cache_home()
    tests_dpath = cache_dpath / 'pdpipe' / 'tests'
    tests_dpath = str(tests_dpath)
    os.makedirs(tests_dpath, exist_ok=True)
    return tests_dpath
