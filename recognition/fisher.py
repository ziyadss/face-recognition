import pickle

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from .constants import CLASSIFIER_PATH


class FisherRecognizer:
    def __init__(self):
        self.scaler: StandardScaler = StandardScaler()
        self.pca: PCA = PCA(n_components=0.95, svd_solver="full")
        self.lda: LinearDiscriminantAnalysis = LinearDiscriminantAnalysis(solver="svd")

    def load(self, path: str = CLASSIFIER_PATH):
        with open(path, "rb") as fd:
            self.scaler, self.pca, self.lda = pickle.load(fd)

    def dump(self, path: str = CLASSIFIER_PATH):
        with open(path, "wb") as fd:
            pickle.dump((self.scaler, self.pca, self.lda), fd)

    def fit(self, X: list, y: list):
        # Standardize the data
        X_standardized = self.scaler.fit_transform(X)

        # Perform PCA to reduce the dimensionality of the data
        X_pca = self.pca.fit_transform(X_standardized)

        # Perform Linear Discriminant Analysis on the reduced data
        self.lda.fit(X_pca, y)

    def predict(self, X: list):
        X_standardized = self.scaler.transform(X)
        X_pca = self.pca.transform(X_standardized)
        return self.lda.predict(X_pca)

    def score(self, X: list, y: list):
        X_standardized = self.scaler.transform(X)
        X_pca = self.pca.transform(X_standardized)
        return self.lda.score(X_pca, y)
