"""Configuring pytest for pdpipe tests."""

import os
import sys
from pathlib import Path

from xdg import BaseDirectory
import pytest
import warnings


# make
sys.path.append(os.path.join(os.path.dirname(__file__), "helpers"))

doctest_global_setup = '''
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*The `disp` and `iprint` options of the L-BFGS-B solver.*",
    category=DeprecationWarning,
)
'''


@pytest.fixture(scope="session", autouse=False)
def pdpipe_tests_dir_path() -> str:
    cache_dpath = BaseDirectory.xdg_cache_home  # This is a string
    tests_dpath = Path(cache_dpath) / "pdpipe" / "tests"
    os.makedirs(tests_dpath, exist_ok=True)
    return str(tests_dpath)
