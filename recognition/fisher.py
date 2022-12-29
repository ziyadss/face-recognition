import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


class fisherfaces:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95, svd_solver="full")
        self.lda = LinearDiscriminantAnalysis(solver="eigen")

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> None:
        # Standardize the data
        X_standardized = self.scaler.fit_transform(X)

        # Perform PCA to reduce the dimensionality of the data
        X_pca = self.pca.fit_transform(X_standardized)

        # Perform Linear Discriminant Analysis on the reduced data
        return self.lda.fit_transform(X_pca, y)


class fisherfaces2:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10, svd_solver="full")
        self.lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Standardize the data
        X_standardized = self.scaler.fit_transform(X)

        # Perform PCA to reduce the dimensionality of the data
        X_pca = self.pca.fit_transform(X_standardized)

        # Perform Linear Discriminant Analysis on the reduced data
        return self.lda.fit_transform(X_pca, y)

    def project(self, X: np.ndarray) -> np.ndarray:
        X_standardized = self.scaler.transform(X)
        X_pca = self.pca.transform(X_standardized)
        print(X_pca.shape)
        print(self.eigenvectors.shape)
        return X_pca @ self.eigenvectors

    @property
    def eigenvectors(self) -> np.ndarray:
        print(f"lda: {self.lda.scalings_.shape}")
        print(f"pca: {self.pca.components_.shape}")
        print(f"dot: {(self.lda.scalings_ * self.pca.components_[:,:10]).shape}")
        return self.lda.scalings_ * self.pca.components_[:, :10]
