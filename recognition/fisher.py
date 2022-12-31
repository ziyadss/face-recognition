import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .constants import CLASSIFIER_PATH
# from .pca import PCA


class FisherRecognizer:
    def __init__(self) -> None:
        self.pca = PCA(n_components=0.95)
        self.lda = LinearDiscriminantAnalysis(solver="svd")
        self.mean = None
        self.std = None

    def load(self, path: str = CLASSIFIER_PATH):
        with open(path, "rb") as fd:
            self.pca, self.lda, self.mean, self.std = pickle.load(fd)

    def dump(self, path: str = CLASSIFIER_PATH):
        with open(path, "wb") as fd:
            pickle.dump((self.pca, self.lda, self.mean, self.std), fd)

    def scale(self, X: list) -> list:
        if self.mean is None or self.std is None:
            # Calculate the mean of each column
            self.mean = np.mean(X, axis=0)

            # Calculate the standard deviation of each column
            self.std = np.std(X, axis=0)

        # Standardize the data
        X_standardized = (X - self.mean) / self.std  # type: ignore

        return X_standardized

    def fit(self, X: list, y: list):
        # Standardize the data
        # X_standardized = self.scaler.fit_transform(X_equalized)
        print("Standardizing data...")
        X_standardized = self.scale(X)

        # Perform PCA to reduce the dimensionality of the data
        print("Performing PCA...")
        X_pca = self.pca.fit_transform(X_standardized)

        # Perform Linear Discriminant Analysis on the reduced data
        print("Performing LDA...")
        self.lda.fit(X_pca, y)

    def predict(self, X: list):
        X_standardized = self.scale(X)
        X_pca = self.pca.transform(X_standardized)
        return self.lda.predict(X_pca)

    def score(self, X: list, y: list):
        X_standardized = self.scale(X)
        X_pca = self.pca.transform(X_standardized)
        return self.lda.score(X_pca, y)
