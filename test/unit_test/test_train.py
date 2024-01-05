import os
import unittest
from turtle import mode

import housing.run_script as path
from housing import train as train

args = path.parse_args()
rootpath = train.get_path()


class TestTrain(unittest.TestCase):
    def test_load_data(self):
        train_X, train_y = train.load_data(rootpath + args.inputpath)
        self.assertTrue(len(train_X) == len(train_y))
        self.assertTrue(len(train_y.shape) == 1)

    def test_save_model(self):
        models = train.model_names
        for i in models:
            self.assertFalse(os.path.isdir(f"{rootpath}{args.outputpath}/{i}"))


if __name__ == "__main__":
    unittest.main()
