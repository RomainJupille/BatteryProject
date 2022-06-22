from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError


class CustomStandardScaler(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0, ddof=0)
        return self

    def transform(self, X, y=None):
        """ standard score: z = (x - u) / s """
        if not (hasattr(self, "means") and hasattr(self, "stds")):
            raise NotFittedError("This CustomStandardScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        z = (X - self.means) / self.stds
        return z

    def inverse_transform(self, X, y=None):
        if not (hasattr(self, "means") and hasattr(self, "stds")):
            raise NotFittedError("This CustomStandardScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        return X * self.stds + self.means
