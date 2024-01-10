import os
import unittest

import pandas as pd
from housing import ingest_data as data
from housing import run_script as start

args = start.parse_args()
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = args.datapath
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
rootpath = data.get_path()


class Testutils(unittest.TestCase):
    def test_parse_args(self):
        self.assertTrue(args.datapath == "data/raw/housing")
        self.assertTrue(args.dataprocessed == "data/processed")
        self.assertTrue(args.inputpath == "data/processed/")
        self.assertTrue(args.modelpath == "artifacts")
        self.assertTrue(args.outputpath == "artifacts")
        self.assertTrue(args.log_level == "DEBUG")
        self.assertFalse(args.no_console_log)
        self.assertTrue(args.log_path == rootpath + "logs/logs.log")


if __name__ == "__main__":
    unittest.main()
