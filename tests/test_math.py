from typing import Tuple
from unittest import TestCase

import numpy as np

from unclog.math import logtrace, bincount_batch


class MathTest(TestCase):
    def test_logtrace_float32(self):
        arr = np.random.uniform(0, 10, (100, 100)).astype(np.float32)
        out = logtrace(arr)

    def test_logtrace_float64(self):
        arr = np.random.uniform(0, 10, (100, 100)).astype(np.float64)
        out = logtrace(arr)

    def test_bincount_batch_int32(self):
        arr = np.random.randint(0, 10, (100, 100)).astype(np.int32)
        out = bincount_batch(arr)

    def test_bincount_batch_int64(self):
        arr = np.random.randint(0, 10, (100, 100)).astype(np.int64)
        out = bincount_batch(arr)

if __name__ == "__main__":
    import unittest

    unittest.main()
