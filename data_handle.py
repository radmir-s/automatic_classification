from scipy.io import loadmat
from sklearn.decomposition import PCA
import pandas as pd

def loadshape(path, align=True):
    s = loadmat(path)['shape']

    if align:
        s = s - s.mean(axis=0)
        pca = PCA(3)
        s = pca.fit_transform(s)   

    return s

def unpack_a2p(csv_path):
    df = pd.read_csv(csv_path).drop(['Unnamed: 0'],axis=1)
    dfs = df.loc[:,['shape', 'shape_class', 'prototype', 'dist']]

    protos = dfs.prototype.unique()

    dfps = list()
    for proto in protos:
        pcode = proto.removeprefix('shape_')
        dfp = dfs[dfs.prototype == proto]
        dfp = dfp.set_index('shape').sort_index()
        dfp = dfp.rename(index=lambda x: x.removeprefix('shape_'))
        dfp = dfp.rename(columns={'dist':pcode})
        dfp = dfp.loc[:, [pcode]]
        dfps.append(dfp)

    dfc = dfp.shape_class
    a2p = pd.concat(dfps, axis=1)

    X = a2p.to_numpy()
    y = dfc.values

    return X, y

