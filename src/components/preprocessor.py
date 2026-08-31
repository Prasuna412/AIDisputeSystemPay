import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import spacy

# Load your spaCy model
# Adjust the model as per your specific needs
nlp = spacy.load('en_core_web_lg')

# Custom transformer for text cleaning


class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.url_pattern = re.compile(
            r'https?://[\w.-]+(?:\.[\w.-]+)+(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.tolist()
        if isinstance(X, list):
            X_clean = []
            for text in X:
                text = re.sub(r'[^\w\s\']', ' ', text)
                text = re.sub(r' +', ' ', text)
                text = re.sub(self.url_pattern, ' ', text)
                text = text.strip().lower()
                X_clean.append(text)
            return X_clean  # Return a NumPy array of cleaned texts
        else:
            raise ValueError(
                "Input should be a list of strings or a pandas Series")


class TextVectorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([nlp(text).vector for text in X])
