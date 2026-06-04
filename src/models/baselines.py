from sklearn.decomposition import IncrementalPCA
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import numpy as np
from typing import Literal

from pygod.detector import DOMINANT


class PCABaseline:
    def __init__(self, n_components=0.95, random_state=42):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)

    def fit(self, x_train):
        self.pca.fit(x_train)
        return self

    def score(self, x_eval):
        z = self.pca.transform(x_eval)

        x_eval_space = self.pca.inverse_transform(z)
        scores = np.mean((x_eval - x_eval_space) ** 2, axis=1)
        return scores


class IsolationForestBaseline:
    def __init__(
        self,
        max_samples: float | Literal["auto"] = "auto",
        random_state=42,
        contamination: float | str = "auto",
    ):
        self.max_samples = max_samples
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.clf = IsolationForest(
            max_samples=max_samples,
            random_state=random_state,
            contamination=contamination,
            n_jobs=-1,
        )

    def fit(self, x_train):
        self.clf.fit(x_train)
        return self

    def score(self, x_eval):
        scores = -self.clf.score_samples(x_eval)
        return scores
