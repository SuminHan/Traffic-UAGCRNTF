import numpy as np
import tensorflow as tf

class Empty:
    def __init__(self):
        self.output_channel = 1

    def fit(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X)
        
    def transform(self, X):
        return X

    def inverse_transform(self, X_norm):
        return X_norm

class ZScoreNormalization:
    def __init__(self, mean=None, std=None, each=False):
        self.mean = mean
        self.std = std
        self.each = each
        self.output_channel = 1

    def fit(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X)
        
    def transform(self, X):
        assert len(X.shape) == 4
        return (X - self.mean) / self.std

    def inverse_transform(self, X_norm):
        assert len(X_norm.shape) == 4
        return X_norm * self.std + self.mean
