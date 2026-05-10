# 3 classes
from sklearn.decomposition import IncrementalPCA
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


class PCABaseline:
    def __init__(self, n_components=2, random_state=42):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
    
    def fit(self, x_train):
        x_train_scale = self.scaler.fit_transform(x_train)
        self.pca.fit(x_train_scale)
        return self
    
    def score(self, x_eval):
        x_eval_scale = self.scaler.transform(x_eval)
        z = self.pca.transform(x_eval_scale)
        
        x_eval_space = self.pca.inverse_transform(z)
        scores = np.mean((x_eval_scale - x_eval_space) ** 2, axis=1)
        return scores
    
    
    
class IsolationForestBaseline:
    def __init__(self):
        raise NotImplemented
    
class SchreyerAE:
    def __init__(self):
        raise NotImplemented
    