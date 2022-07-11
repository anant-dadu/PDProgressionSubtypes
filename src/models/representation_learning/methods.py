import pandas as pd
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
def apply_dimensionality_reduction(X, how={'NMF': 2}):
    method = list(how)[0]
    n_dim = how[method]
    if method == 'NMF':
        # model =  NMF(n_components=n_dim, init='random', random_state=42, alpha=1, l1_ratio=1)
        model =  NMF(n_components=n_dim, init='nndsvd', random_state=42)
    elif method == 'PCA':
        model = PCA(n_components=n_dim, random_state=42)
    elif method == 'ICA':
        model = FastICA(n_components=n_dim, random_state=42)
    W = model.fit_transform(X.values)
    W = pd.DataFrame(W, index=X.index, columns=['latent_weight{}'.format(i) for i in range(1, n_dim+1)])
    W['model_name'] = [method] * len(W)
    return W, model

from sklearn.decomposition import NMF, PCA, FastICA
import umap


def apply_dimensionality_reduction_te(X, method = 'NMF', n_dim = 2):
    if method == 'NMF':
        model = NMF(n_components=n_dim, init='nndsvd', random_state=42)
    elif method == 'PCA':
        model = PCA(n_components=n_dim, random_state=42)
    elif method == 'ICA':
        model = FastICA(n_components=n_dim, random_state=42)
    elif method == 'UMAP':
        model = umap.UMAP(n_components=n_dim)
    W = model.fit_transform(X)
    return model, W
