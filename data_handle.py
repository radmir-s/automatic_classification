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

def unpack_a2p_dq(csv_path, diff=False):
    df = pd.read_csv(csv_path)

    if diff:
        qcols = df.columns[df.columns.str.startswith('q')].to_list()[::-1]
        for k in range(9):
            df[qcols[k]] = df[qcols[k]] - df[qcols[k+1]]
    
    dfps = list()

    for proto in df.prototype.unique():
        dfp = df[df.prototype == proto]
        dfp = dfp.set_index('shape').sort_index()
        y = dfp.shape_class.values
        dfp = dfp.rename(columns={f'q{k/10:01}':f'{proto}.q{k/10:01}' for k in range(1,11)})
        dfp = dfp.loc[:,dfp.columns.str.startswith(str(proto))]
        dfps.append(dfp)
    
    a2p = pd.concat(dfps, axis=1)
    X = a2p.values
    
    return X, y