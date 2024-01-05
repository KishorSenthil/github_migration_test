import os
import unittest
from turtle import mode

import housing.run_script as path
from housing import score as score

args = path.parse_args()
rootpath = score.get_path()


class TestTrain(unittest.TestCase):
    def test_load_data(self):
        test_X, test_y = score.load_data(rootpath + args.dataprocessed)
        self.assertTrue(len(test_X) == len(test_y))
        self.assertTrue(len(test_y.shape) == 1)

    def test_load_models(self):
        models = score.load_models(rootpath + args.modelpath)
        self.assertTrue(len(models) == 4)


if __name__ == "__main__":
    unittest.main()
