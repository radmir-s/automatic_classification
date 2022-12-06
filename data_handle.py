from scipy.io import loadmat
from sklearn.decomposition import PCA

def loadshape(path, align=True):
    s = loadmat(path)['shape']

    if align:
        s = s - s.mean(axis=0)
        pca = PCA(3)
        s = pca.fit_transform(s)   

    return s