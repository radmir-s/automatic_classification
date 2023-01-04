from scipy.io import loadmat
from sklearn.decomposition import PCA
import pandas as pd

def loadshape(path, res = '',align=True):
    if res:
        s = loadmat(path)[res]
    else:
        s = loadmat(path)['shape']

    if align:
        s = s - s.mean(axis=0)
        pca = PCA(3)
        s = pca.fit_transform(s)   

    return s

def unpack_a2p(csv_path):
    dfs = pd.read_csv(csv_path).drop(['Unnamed: 0'],axis=1)
    protos = dfs.prototype.unique()
    protos.sort()

    dfps = list()
    for proto in protos:
        dfp = dfs[dfs.prototype == proto]
        dfp = dfp.set_index('shape').sort_index()
        dfc = dfp.shape_class
        dfp = dfp.rename(columns={'dist':proto})
        dfp = dfp.loc[:, [proto]]
        dfps.append(dfp)

    a2p = pd.concat(dfps, axis=1)
    X = a2p.values
    y = dfc.values

    return X, y

