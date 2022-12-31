import numpy as np


class PCA:
    def __init__(
        self, n_components: int | float = 0.95, order_components: bool = False
    ) -> None:
        self.n_components = n_components
        self.order = order_components
        self.mean = None
        self.eigenvalues = None
        self.eigenvectors = None

    def fit_transform(self, X: list) -> list:
        # Calculate the mean of each column
        self.mean = np.mean(X, axis=0)

        # Center the data
        X_centered = X - self.mean

        # Calculate the covariance matrix
        cov = np.cov(X_centered, rowvar=False)

        # Calculate the eigenvalues and eigenvectors of the covariance matrix using SVD
        u, s, vh = np.linalg.svd(cov)

        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        vh *= signs[:, np.newaxis]

        if self.order:
            # Sort the eigenvalues and eigenvectors in descending order
            idx = s.argsort()[::-1]
            s = s[idx]
            vh = vh[:, idx]

        self.eigenvalues = s
        self.eigenvectors = vh

        # Calculate the number of components to keep
        if isinstance(self.n_components, float):
            total = np.sum(s)
            current = 0
            for i, eigenvalue in enumerate(s):
                current += eigenvalue
                if current / total >= self.n_components:
                    self.n_components = i + 1
                    break

        # Calculate the projection matrix
        projection_matrix = vh[: self.n_components]

        # Project the data onto the lower dimensional space
        X_pca = np.dot(X_centered, projection_matrix.T)

        return X_pca

    def transform(self, X: list) -> list:
        if self.mean is None or self.eigenvalues is None or self.eigenvectors is None:
            self.fit_transform(X)

        # Center the data
        X_centered = X - self.mean  # type: ignore

        # Calculate the projection matrix
        projection_matrix = self.eigenvectors[: self.n_components]  # type: ignore

        # Project the data onto the lower dimensional space
        X_pca = np.dot(X_centered, projection_matrix.T)

        return X_pca
