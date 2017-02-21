"""Testing basic pipline stages."""

import math
import collections
from unittest import TestCase

import pandas as pd

from pdpipe.basic_stages import (
    ColDrop,
    Bin
)

__author__ = "Shay Palachy"
__copyright__ = "Shay Palachy"
__license__ = "MIT"


class TestColDrop(TestCase):
    """Testing the ColDrop pipline stage."""

    def test_coldrop_1(self):
        """Testing the ColDrop pipline stage."""
        df = pd.DataFrame(
            data=[[1, 'a'], [2, 'b']],
            index=[1, 2],
            columns=['num', 'char']
        )
        self.assertEqual(
            collections.Counter(df.columns),
            collections.Counter(['num', 'char']))
        self.assertTrue('num' in df.columns)
        stage = ColDrop('num')
        df = stage.apply(df)
        self.assertTrue('num' not in df.columns)
        self.assertTrue('char' in df.columns)


class TestBin(TestCase):
    """Testing the Bin pipline stage."""

    def test_bin_1(self):
        """Testing the Bin pipline stage."""
        binner = Bin._get_col_binner([0, 5])
        self.assertEqual(binner(-math.inf), '< 0')
        self.assertEqual(binner(-4), '< 0')
        self.assertEqual(binner(0), '0-5')
        self.assertEqual(binner(1), '0-5')
        self.assertEqual(binner(4.99), '0-5')
        self.assertEqual(binner(5), '5 ≤')
        self.assertEqual(binner(232), '5 ≤')
        self.assertEqual(binner(math.inf), '5 ≤')


