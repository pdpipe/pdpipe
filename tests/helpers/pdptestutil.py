"""Shared testing functionalities for pdpipe."""

import os
import random


def random_pickle_path(pdpipe_tests_dpath: str) -> str:
    n = random.randint(1, 999999)
    return os.path.join(
        pdpipe_tests_dpath,
        f'pdpipe_pickled_obj_{n}',
    )
