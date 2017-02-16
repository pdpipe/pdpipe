"""Testing basic pipline stages."""

import math

from unittest import TestCase

from pdpipe.basic_stages import (
    Bin
)

__author__ = "Shay Palachy"
__copyright__ = "Shay Palachy"
__license__ = "MIT"


class TestBin(TestCase):
    """Testing the Bin pipline stage."""

    def test_find_point_1(self):
        """Testing the Bin pipline stage."""
        print("Bin test 1")
        binner = Bin._get_col_binner([0, 5])
        self.assertEqual(binner(-math.inf), '< 0')
        self.assertEqual(binner(-4), '< 0')
        self.assertEqual(binner(0), '0-5')
        self.assertEqual(binner(1), '0-5')
        self.assertEqual(binner(4.99), '0-5')
        self.assertEqual(binner(5), '5 ≤')
        self.assertEqual(binner(232), '5 ≤')
        self.assertEqual(binner(math.inf), '5 ≤')


