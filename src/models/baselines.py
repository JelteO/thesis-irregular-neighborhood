# baselines.py
# Two simple tabular baselines (PCA & IF)
# in: feature matrix (n_entries, 618), fitted on train rows only
# out: one anomaly score per entry, higher means more anomalous
# Both models follow same fit/score interface as the other baselines
# So the experiment cade can treat every model the same


from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import numpy as np
from typing import Literal


class PCABaseline:
    def __init__(self, n_components=0.95, random_state=42):
        # 0.95 keeps enough components to explain 95% of the variance in the train data
        # Those components describe the common pattern
        # Anything more rare then that cannot be reconstructed, which
        # is what makes the back-projection error an anomaly score
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=random_state)

    def fit(self, x_train):
        self.pca.fit(x_train)
        return self

    def score(self, x_eval):
        # PCA score in two steps
        # 1, transorm 618 features down onto the components; drops everything
        # components do not cover
        # 2, inverse transform projects this back up to 618 features
        z = self.pca.transform(x_eval)
        x_eval_space = self.pca.inverse_transform(z)

        # reconstruction error is distance between entry and
        # the projection from the component space
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
        # sklearn return higher values for normal points, so the sign is
        # flipped to get a score that increases with how anomalous a point is
        scores = -self.clf.score_samples(x_eval)
        return scores
