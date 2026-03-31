import unittest
import mlflow

import os 


class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
    MLFLOW_TRACKING_USERNAME = os.getenv()
    MLFLOW_TRACKING_PASSWORD = os.getenv()