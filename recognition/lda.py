from typing import Any

import numpy as np


# replacement for sklearn.discriminant_analysis.LinearDiscriminantAnalysis
class LDA:
    def __init__(self) -> None:
        self.classes = None
        self.means = None

    def fit(self, X: list | np.ndarray, y: list | np.ndarray) -> None:
        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)

        mean = np.mean(X, axis=0)

        self.classes = np.array(sorted(np.unique(y)))

        # Calculate the class means
        classes, yc = np.unique(y, return_inverse=True)
        self.means = np.zeros((classes.shape[0], X.shape[1]))

        for i, label in enumerate(classes):
            self.means[i] = np.mean(X[yc == i], axis=0)

        # self.means = []
        # for i in range(n_classes):
        #     self.means.append(np.mean(X[y == self.classes[i]], axis=0))
        # self.means = np.array(self.means)

        # Calculate the within-class scatter matrix
        S_w = np.zeros((X.shape[1], X.shape[1]))
        for label in np.unique(y):
            X_label = X[y == label]
            S_w += np.dot(X_label.T, X_label)

        # Calculate the between-class scatter matrix
        S_b = np.zeros((X.shape[1], X.shape[1]))
        for i, mean_i in enumerate(self.means):
            n_i = X[y == i].shape[0]
            mean_diff = (mean_i - mean).reshape(X.shape[1], 1)
            S_b += n_i * np.dot(mean_diff, mean_diff.T)

        # Calculate the eigenvalues and eigenvectors of the generalized eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))

        # Sort the eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Calculate the projection matrix
        self.projection_matrix = eigenvectors

    def transform(self, X: list) -> list:
        # Project the data onto the lower dimensional space
        X_lda = np.dot(X, self.projection_matrix)

        return X_lda

    def predict(self, X: list) -> Any:
        if self.means is None:
            raise Exception("Model not trained")

        # Project the data onto the lower dimensional space
        X_lda = np.dot(X, self.projection_matrix)

        # Calculate the Mahalanobis distance between each sample and each class mean
        distances = []
        for mean in self.means:
            distance = np.sum((X_lda - mean) ** 2, axis=1)
            distances.append(distance)
        distances = np.array(distances)

        # Return the class with the smallest distance
        return np.argmin(distances, axis=0)

    def score(self, X: list, y: list) -> float:
        if self.means is None:
            raise Exception("Model not trained")

        # Project the data onto the lower dimensional space
        X_lda = np.dot(X, self.projection_matrix)

        # Calculate the Mahalanobis distance between each sample and each class mean
        distances = []
        for mean in self.means:
            distance = np.sum((X_lda - mean) ** 2, axis=1)
            distances.append(distance)
        distances = np.array(distances)

        # Return the accuracy
        return float(np.mean(np.argmin(distances, axis=0) == y))

    def predict_proba(self, X: list) -> list:
        if self.means is None:
            raise Exception("Model not trained")

        # Project the data onto the lower dimensional space
        X_lda = np.dot(X, self.projection_matrix)

        # Calculate the Mahalanobis distance between each sample and each class mean
        distances = []
        for mean in self.means:
            distance = np.sum((X_lda - mean) ** 2, axis=1)
            distances.append(distance)
        distances = np.array(distances)

        # Calculate the probability of each class
        probabilities = np.exp(-distances) / np.sum(np.exp(-distances), axis=0)

        return probabilities

    def predict_log_proba(self, X: list) -> list:
        if self.means is None:
            raise Exception("Model not trained")

        # Project the data onto the lower dimensional space
        X_lda = np.dot(X, self.projection_matrix)

        # Calculate the Mahalanobis distance between each sample and each class mean
        distances = []
        for mean in self.means:
            distance = np.sum((X_lda - mean) ** 2, axis=1)
            distances.append(distance)
        distances = np.array(distances)

        # Calculate the log probability of each class
        log_probabilities = -distances - np.log(np.sum(np.exp(-distances), axis=0))

        return log_probabilities
