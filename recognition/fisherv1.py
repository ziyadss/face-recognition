import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC


def pca(X: np.ndarray, k: int, staderdize: bool = False) -> np.ndarray:
    """
    Reduce the dimensionality of the input data using PCA.

    Parameters:
    - X: A 2D array of size (n_samples, n_features)
    - k: The number of PCA features to keep

    Returns:
    - features: A 2D array of size (n_features, k) containing the PCA features
    """
    # Standardize the input data by subtracting the mean and dividing by the standard deviation
    if staderdize:
        mean, std = np.mean(X, axis=0), np.std(X, axis=0)
        X = (X - mean) / std

    # Compute the covariance matrix of the standardized data
    cov = np.cov(X, rowvar=False)

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort the eigenvectors in the same order as the eigenvalues
    eigenvectors = eigenvectors[:, eigenvalues.argsort()[::-1]]

    # Select the top k eigenvectors and project the data onto the top k eigenvectors using matrix multiplication
    eigenvectors = abs(eigenvectors)
    print(eigenvectors)
    return eigenvectors[:, :k]


def lda(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """
    Reduce the dimensionality of the input data using LDA.

    Parameters:
    - X: A 2D array of size (n_samples, n_features)
    - y: A 1D array of size (n_samples) containing the labels of the samples
    - k: The number of LDA features to keep

    Returns:
    - features: A 2D array of size (n_features, k) containing the LDA features
    """
    # Standardize the input data by subtracting the mean and dividing by the standard deviation
    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    standardized = (X - mean) / std

    # Compute the within-class scatter matrix and the between-class scatter matrix
    Sw = np.zeros((X.shape[1], X.shape[1]))
    Sb = np.zeros((X.shape[1], X.shape[1]))
    classes = np.unique(y)
    for c in classes:
        Xc = standardized[y == c]
        mean_c = np.mean(Xc, axis=0)
        # slower alternative computation for the scatter matrices
        # Sw += np.cov(Xc, rowvar=False)
        # Sb += Xc.shape[0] * np.outer(mean_c - mean, mean_c - mean)
        Sw += np.abs(np.dot((Xc - mean_c).T, (Xc - mean_c)))
        Sb += np.abs(np.dot((mean_c - mean).T, (mean_c - mean)))

    # Compute the eigenvalues and eigenvectors of the scatter matrices
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw) @ Sb)

    # Sort the eigenvectors in the same order as the eigenvalues
    eigenvectors = eigenvectors[:, eigenvalues.argsort()[::-1]]

    # Select the top k eigenvectors and project the data onto the top k eigenvectors using matrix multiplication
    eigenvectors = abs(eigenvectors)
    print(eigenvectors)
    return eigenvectors[:, :k]


def extract_fisherfaces(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """
    Extract Fisherface features from the input data.

    Parameters:
    - X: A 2D array of size (n_samples, n_features)
    - y: A 1D array of size (n_samples) containing the labels of the samples
    - k: The number of Fisherface features to extract

    Returns:
    - features: A 2D array of size (n_features, k) containing the extracted Fisherface features
    """
    # Standardize the input data by subtracting the mean and dividing by the standard deviation
    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    standardized = (X - mean) / std

    # Compute the PCA features of the input data
    # PCA is used to decorrelate the features and reduce the dimensionality of the data
    print("Computing PCA features...")
    pca_eigenvectors = pca(X, len(y) - len(np.unique(y)))
    print("Done.")

    # Compute the LDA features of the PCA features
    # LDA is used to maximize the between-class separation and minimize the within-class variance
    print("Computing LDA features...")
    lda_eigenvectors = lda((standardized @ pca_eigenvectors).reshape(-1, 1), y, k)
    print("Done.")

    eigenvectors = abs(pca_eigenvectors @ lda_eigenvectors).reshape(-1, 1)
    features = standardized @ eigenvectors

    print(eigenvectors)
    print(features)

    return features


def direct_fisherfaces(X, y, n_components):
    """
    Computes the Fisherfaces of the given data.

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        The data matrix. Each row corresponds to a sample.
    y : numpy array, shape (n_samples,)
        The labels of the samples.
    n_components : int
        The number of Fisherfaces to compute.

    Returns
    -------
    W : numpy array, shape (n_components, n_features)
        The Fisherfaces. Each row corresponds to a Fisherface.
    """
    n_samples, n_features = X.shape

    # Center the data
    X = X - np.mean(X, axis=0) / np.std(X, axis=0)

    # Compute the between-class scatter matrix
    Sb = np.zeros((n_features, n_features))
    for c in np.unique(y):
        mean_vector = np.mean(X[y == c], axis=0)
        mean_vector = mean_vector.reshape(n_features, 1)
        Sb += mean_vector.dot(mean_vector.T)
    Sb -= n_samples * np.mean(X, axis=0).reshape(n_features, 1).dot(
        np.mean(X, axis=0).reshape(1, n_features)
    )

    # Compute the within-class scatter matrix
    Sw = np.zeros((n_features, n_features))
    for c in np.unique(y):
        X_c = X[y == c]
        X_c_centered = X_c - np.mean(X_c, axis=0)
        Sw += X_c_centered.T.dot(X_c_centered)

    # Compute the eigenvectors and eigenvalues of the matrix Sw^{-1}Sb
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

    # Sort the eigenvectors and eigenvalues in decreasing order of eigenvalues
    eigenvalues_sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues_sorted_indices]
    eigenvectors = eigenvectors[:, eigenvalues_sorted_indices]

    # Compute the Fisherfaces
    W = eigenvectors[:, :n_components]

    return abs(W)


def reduce_dimensionality(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """
    Reduce the dimensionality of the input data using feature selection.

    Parameters:
    - X: A 2D array of size (n_samples, n_features)
    - y: A 1D array of size (n_samples) containing the labels of the samples
    - k: The number of features to select

    Returns:
    - reduced: A 2D array of size (n_samples, k) containing the selected features
    """
    print("Reducing dimensionality...")
    # Use the SelectKBest function to select the top k features
    # make sure everything is real
    selector = SelectKBest(k=k)
    reduced = selector.fit_transform(np.abs(X), y)

    print("Done.")
    return reduced


def project_faces(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """
    Project the input data onto the Fisherface features.

    Parameters:
    - X: A 2D array of size (n_samples, n_features)
    - y: A 1D array of size (n_samples) containing the labels of the samples
    - k: The number of Fisherface features to use

    Returns:
    - projected: A 2D array of size (n_samples, k) containing the projected data
    """
    # Extract the Fisherface features
    fisherfaces = extract_fisherfaces(X, y, k)

    # Reduce the dimensionality of the fisherfaces
    fisherfaces = reduce_dimensionality(fisherfaces, y, k)

    # Standardize the input data by subtracting the mean and dividing by the standard deviation
    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    standardized = (X - mean) / std

    # Project the data onto the Fisherface features using matrix multiplication
    projected = standardized @ fisherfaces

    return projected


def train_classifier(X: np.ndarray, y: np.ndarray) -> SVC:
    """
    Train a classifier on the input data.

    Parameters:
    - X: A 2D array of size (n_samples, n_features)
    - y: A 1D array of size (n_samples) containing the labels of the samples

    Returns:
    - clf: A trained classifier
    """
    print("Training classifier...")
    # Use an SVM classifier
    clf = SVC()

    # Train the classifier on the input data
    clf.fit(X, y)

    print("Done.")
    return clf


def predict(clf: SVC, X: np.ndarray) -> np.ndarray:
    """
    Use the classifier to predict the labels of new samples.

    Parameters:
    - clf: A trained classifier
    - X: A 2D array of size (n_samples, n_features)

    Returns:
    - y_pred: A 1D array of size (n_samples) containing the predicted labels
    """
    # Use the classifier to predict the labels of the new samples
    y_pred = clf.predict(X)

    return y_pred
