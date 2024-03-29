from scipy.io import loadmat
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def vtk2numpy_dir(vtkdir, npydir):
    vtkfiles = {file.removesuffix('.vtk'):os.path.join(vtkdir, file) for file in os.listdir(vtkdir) if file.endswith('vtk')}
    for filename, vtkpath in vtkfiles.items():
        points = readvtk(vtkpath)

        np.save(os.path.join(npydir, filename),points)

def readvtk2(vtk_path, align=True, rescale=False):
    with open(vtk_path,'r') as file:
        section = 'header'
        points = []
        polygons = []
        for line in file.readlines():

            if (section == 'header') and line.lower().startswith('points'):
                section = 'points'
                pnum = int(line.split()[1])
                continue
            
            elif (section == 'points') and line.lower().startswith('polygons'):
                section = 'polygons'
                continue

            elif (section == 'polygons') and line.lower().startswith('cell_data'):
                break
            
            match section:
                case 'header':
                    continue
                case 'points':
                    p = list(map(float, line.split()))
                    points.append(p)
                case 'polygons':
                    p = list(map(int, line.split()))
                    polygons.append(p)

    s = np.array(points)
    connections = np.array(polygons)

    if align:
        s = s - s.mean(axis=0)

        pca = PCA(3).fit(s)
        rotation = pca.components_.T if np.linalg.det(pca.components_)>0 else -pca.components_.T
        s = s @ rotation
        
        if rescale:
            s /= pca.singular_values_

    return s, connections

def readvtk(vtk_path, align=True, rescale=False):
    with open(vtk_path,'r') as file:
        lines = file.readlines()

    scan = 'header'
    for line in lines:

        match scan:
            case 'header':
                if line.lower().startswith('points'):
                    numpoints = int(line.split()[1])
                    points = np.zeros((numpoints, 3))
                    scan = 'points'
                    ind = 0

            case 'points':
                if line.lower().startswith('triangle_strips'):
                    scan = 'exit'
                    break
                numbers = tuple(map(float,line.split()))
                assert len(numbers)%3 == 0 
                for k in range(0, len(numbers), 3):
                    points[ind] = numbers[k:k+3]
                    ind += 1

    s = points
    if align:
        s = s - s.mean(axis=0)
        pca = PCA(3)
        s = pca.fit_transform(s)   
        
        if rescale:
            s /= pca.singular_values_
        
    return s 


def epsnet(X, eps, L='L2', pick='max'):
    representers = []
    left = np.ones(X.shape[0], dtype=bool)

    # pick distance from here L1 vs L2
    match L:
        case 'L1':
            dist = np.sum(np.abs(X.reshape(1,-1,X.shape[1]) - X.reshape(-1,1,X.shape[1])),axis=-1)
        case 'L2':
            dist = np.sqrt(np.sum((X.reshape(1,-1,X.shape[1]) - X.reshape(-1,1,X.shape[1]))**2,axis=-1))
        case _:
            raise KeyError("The 'dist' parameter must be either 'L1' or  'L2'")

    connections = dist<eps
    neigh_sizes = []

    while np.any(left):
        match pick:
            case 'max':
                next_repr_ind = np.argmax(np.sum(connections, axis=1))
            case 'random':
                next_repr_ind = np.random.choice(np.argwhere(left).flatten())
            case _:
                raise KeyError("The 'pick' parameter must be either 'max' or  'random'")
        
        representers.append(next_repr_ind)
        neighbours = connections[next_repr_ind]
        neigh_sizes.append(sum(neighbours))

        left[neighbours] = False
        connections[~left] = False
        connections[:, ~left] = False

    representer_power = np.sum(dist[representers] < eps, axis=1)
    redundancy = np.sum(dist[representers] < eps, axis=0)

    return representers, representer_power, redundancy

def loadshape(path, res = '', align=True, rescale=False):
    if res:
        s = loadmat(path)[res]
    else:
        s = loadmat(path)['shape']

    if align:
        s = s - s.mean(axis=0)
        pca = PCA(3)
        s = pca.fit_transform(s)   
        
        if rescale:
            s /= pca.singular_values_

    return s

def unpack_d2d_2(csv_path, metric):
    df = pd.read_csv(csv_path).drop('Unnamed: 0', axis=1)
    shapes1 = sorted(df.shape1.unique())
    shapes2 = sorted(df.shape2.unique())
    df = df.set_index(['shape1', 'shape2'])

    data = dict()
    for a in shapes1:
        data[a] = df.loc[a].loc[shapes2][metric].values.flatten()

    return pd.DataFrame(data, columns=shapes1,index=shapes2)

def unpack_d2d(csv_path):
    df = pd.read_csv(csv_path).drop('Unnamed: 0', axis=1)
    shapes1 = df.shape1.unique()
    shapes2 = df.shape2.unique()
    df = df.set_index(['shape1', 'shape2'])

    data = dict()
    for a in shapes1:
        data[a] = df.loc[a].loc[shapes2].values.flatten()

    return pd.DataFrame(data, columns=shapes1,index=shapes2)


def unpack_a2p(csv_path):
    dfs = pd.read_csv(csv_path).drop(['Unnamed: 0'],axis=1)
    protos = dfs.prototype.unique()
    protos.sort()

    dfps = list()
    for proto in protos:
        dfp = dfs[dfs.prototype == proto]
        dfp = dfp.set_index('shape').sort_index()
        dfp = dfp.rename(columns={'dist':proto})
        dfp = dfp.loc[:, [proto]]
        dfps.append(dfp)

    a2p = pd.concat(dfps, axis=1)
    X = a2p.values

    dfp = dfs[dfs.prototype == protos[0]]
    dfp = dfp.set_index('shape').sort_index()
    dfc = dfp.shape_class    
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
        dfp = dfp.rename(columns={f'q{k/10:01}':f'{proto}.q{k/10:01}' for k in range(1,11)})
        dfp = dfp.loc[:,dfp.columns.str.startswith(str(proto))]
        dfps.append(dfp)
    
    a2p = pd.concat(dfps, axis=1)
    X = a2p.values

    dfp = df[df.prototype == df.prototype[0]]
    dfp = dfp.set_index('shape').sort_index()
    y = dfp.shape_class.values
    
    return X, y