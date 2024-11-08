#!/usr/bin/env python3

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler


class ModelTraining:
    def __init__(self):
        self.dataset = fetch_ucirepo(id=186)
        self.X = self.dataset.data.features
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.y = self.dataset.data.targets
        self.y = self.y.astype({"quality": np.float64})
        self.y.head(20)

    def train_model(self, model, model_name):
        self.model = model
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            random_state=40 + 2,
            test_size=0.1,
        )

        self.model.fit(self.X_train, self.y_train)

    def measure_model(self):
        self.pred = np.reshape(self.model.predict(self.X_test), (-1,))
        self.y_test = self.y_test.values.ravel()
        print(self.y_test[:10])
        print(self.pred[:10])
        return np.array([r2_score(self.y_test, self.pred)], dtype=np.float64)


if __name__ == "__main__":
    MT = ModelTraining()
    MT.train_model(LinearRegression(), "Linear reg")
    print(MT.measure_model())
    # MT.train_model(Lasso(alpha=0.1), "Lasso")
    # print(MT.measure_model())
    # MT.train_model(LinearSVR(max_iter=3000), "")
    # print(MT.measure_model())
