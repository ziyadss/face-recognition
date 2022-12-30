from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


class fisherfaces:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95, svd_solver="full")
        self.lda = LinearDiscriminantAnalysis(solver="svd")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Histogram Equalization
        # X_equalized = histogram_equalization(X)

        # Standardize the data
        # X_standardized = self.scaler.fit_transform(X_equalized)
        X_standardized = self.scaler.fit_transform(X)

        # Perform PCA to reduce the dimensionality of the data
        X_pca = self.pca.fit_transform(X_standardized)

        # Perform Linear Discriminant Analysis on the reduced data
        self.lda.fit(X_pca, y)

    def predict(self, X: np.ndarray) -> Any:
        # X_equalized = histogram_equalization(X)
        # X_standardized = self.scaler.transform(X_equalized)
        X_standardized = self.scaler.transform(X)
        X_pca = self.pca.transform(X_standardized)
        return self.lda.predict(X_pca)
