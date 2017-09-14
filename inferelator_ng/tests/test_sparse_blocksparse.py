import unittest
from .. import sparse_blocksparse
import pandas as pd
import numpy as np


class Test_MT_SBS_regression(unittest.TestCase):

    def test_(self):
        pass

class Test_MT_SBS_OneGene(unittest.TestCase):

    def test_(self):
        pass

class Test_AuxiliaryFunctions(unittest.TestCase):
    X = [np.array([[1,1,1], [0,0,1], [2,2,2]]),
              np.array([[2,2,2], [2,2,2], [3,3,3]])]
    Y = [np.array([3, 1, 6]).reshape(3,1),
              np.array([12, 12, 18]).reshape(3,1)]

    def test_sum_squared_errors_RSSzero(self):
        W = np.array([[1,2], [1,2], [1,2]])
        # for each task separately
        self.assertEqual(sparse_blocksparse.sum_squared_errors(self.X, self.Y, W, 0), 0)
        self.assertEqual(sparse_blocksparse.sum_squared_errors(self.X, self.Y, W, 1), 0)

    def test_ebic_RSSzero(self):
        W = np.array([[1,2], [1,2], [1,2]])
        n_samples = [2, 2]; n_preds = 3; n_tasks = 2
        self.assertAlmostEqual(sparse_blocksparse.ebic(self.X, self.Y, W, n_tasks, n_samples, n_preds), float('-Inf'))
