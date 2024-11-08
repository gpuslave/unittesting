import os, sys
import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unittesting.master import ModelTraining

# models = [
#     (LinearRegression(), "LinearReg"),
#     (LinearSVR(max_iter=2300), "SVR"),
# ]

models = [
    (LinearRegression(), "LinearReg"),
]

ref_float = np.array([np.float64(0.30)], dtype=np.float64)


class TestModelPerf(unittest.TestCase):
    def test_f2(self):
        MT = ModelTraining()
        for model in models:
            MT.train_model(*model)
            measure = MT.measure_model()

            try:
                np.testing.assert_allclose(
                    ref_float,
                    measure,
                    atol=0.05,
                )
            except AssertionError:
                self.assertTrue((measure >= 0.35).all())


if __name__ == "__main__":
    unittest.main()
