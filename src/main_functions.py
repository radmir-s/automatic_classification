import numpy as np
import time
import ot

def emd(s1, s2, vox=5e-2):
    start = time.time()
    cd1 = densify(s1, vox=vox)
    cd2 = densify(s2, vox=vox)

    c1 = cd1[:, :3]
    d1 = cd1[:, 3:].flatten()

    c2 = cd2[:, :3]
    d2 = cd2[:, 3:].flatten()   
                
    M = np.sqrt( ot.dist(c1, c2) )

    T, log = ot.emd(d1, d2, M, log=True)

    end = time.time()
    runtime = end-start

    return np.sum(T*M), runtime, None, T.shape


def sinkhorn(s1, s2, vox=5e-2, reg=5e-3, numItermax=10_000):
    start = time.time()
    cd1 = densify(s1, vox=vox)
    cd2 = densify(s2, vox=vox)

    c1 = cd1[:, :3]
    d1 = cd1[:, 3:].flatten()

    c2 = cd2[:, :3]
    d2 = cd2[:, 3:].flatten()   
                
    M = np.sqrt( ot.dist(c1, c2) )

    T, log = ot.sinkhorn(d1, d2, M, reg=1e-2, method='sinkhorn', log=True, numItermax=numItermax)


    end = time.time()
    runtime = end-start
    
    return np.sum(T*M), runtime, log['niter'], T.shape


def densify2(s, cubenum=None):

    if cubenum is None:
        cubenum = s.shape[0]

    x1, y1, z1 = s.min(axis=0)
    x2, y2, z2 = s.max(axis=0)

    volume = (x2 - x1) * (y2 - y1) * (z2 - z1)
    css = np.cbrt(volume / cubenum) # cube side size

    xx = np.arange(x1 - css/2, x2 + css, css)
    yy = np.arange(y1 - css/2, y2 + css, css)
    zz = np.arange(z1 - css/2, z2 + css, css)

    dimx = len(xx) - 1
    dimy = len(yy) - 1
    dimz = len(zz) - 1

    sx = s[:,0].reshape(1,-1)
    sy = s[:,1].reshape(1,-1)
    sz = s[:,2].reshape(1,-1)

    indx = (xx[:-1].reshape(-1,1) < sx) & (sx < xx[1:].reshape(-1,1))
    indy = (yy[:-1].reshape(-1,1) < sy) & (sy < yy[1:].reshape(-1,1))
    indz = (zz[:-1].reshape(-1,1) < sz) & (sz < zz[1:].reshape(-1,1))

    indxyz = indx.reshape(dimx,1,1,-1) * indy.reshape(1,dimy,1,-1) * indz.reshape(1,1,dimz,-1)
    dens = np.sum(indxyz, axis=3)
    dens = dens/dens.sum() # voxel

    densecloud = list()
    for ixyz in np.argwhere(dens):
        cubeind = indxyz[*ixyz]
        cubepoints = s[cubeind]
        center = cubepoints.mean(axis=0)
        densecloud.append([*center, dens[*ixyz]])

    densecloud = np.array(densecloud)

    return densecloud


def densify(s, **kwargs):
    
    x1, y1, z1 = s.min(axis=0) - 1e-2
    x2, y2, z2 = s.max(axis=0) + 1e-2
    
    if 'vox' in kwargs:
        vox = kwargs['vox']
        xx = np.arange(x1, x2, vox)
        yy = np.arange(y1, y2, vox)
        zz = np.arange(z1, z2, vox)

        dimx = len(xx) - 1
        dimy = len(yy) - 1
        dimz = len(zz) - 1
    elif 'dim' in kwargs:
        dimx, dimy, dimz = kwargs['dim']
        xx = np.linspace(x1,x2,dimx+1)
        yy = np.linspace(y1,y2,dimy+1)
        zz = np.linspace(z1,z2,dimz+1)
    else:
        raise KeyError("Either 'vox' or 'dim' parameters should be defined")

    sx = s[:,0].reshape(1,-1)
    sy = s[:,1].reshape(1,-1)
    sz = s[:,2].reshape(1,-1)

    indx = (xx[:-1].reshape(-1,1) < sx) & (sx < xx[1:].reshape(-1,1))
    indy = (yy[:-1].reshape(-1,1) < sy) & (sy < yy[1:].reshape(-1,1))
    indz = (zz[:-1].reshape(-1,1) < sz) & (sz < zz[1:].reshape(-1,1))

    voxel = np.sum(indx.reshape(dimx,1,1,-1) * indy.reshape(1,dimy,1,-1) * indz.reshape(1,1,dimz,-1), axis=-1 )
    voxel = voxel/voxel.sum()

    x = (xx[:-1] + xx[1:])/2
    y = (yy[:-1] + yy[1:])/2
    z = (zz[:-1] + zz[1:])/2

    histocloud = [[x[indx], y[indy], z[indz], voxel[indx, indy, indz]] for indx, indy, indz in np.argwhere(voxel)]
    histocloud = np.array(histocloud)

    return histocloud


def hausq_dist(x, y, q = np.linspace(0,1,11)):
    w = np.ones_like(q, dtype=float)
    w[0] = 0.5
    w[-1] = 0.5
    quants = hausdorff_quant(x, y, q)

    return np.sum(quants * w)


def hausdorff_quant(x, y, q=np.linspace(0,1,11)):
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray), \
        "x and y have to be numpy arrays"
    assert x.shape[1] == y.shape[1] == 3, \
        "x and y have to be in 3d"
    D = x.reshape((-1,1,3)) - y.reshape((1,-1,3))
    D2 = np.sum(D**2,axis=2)
    Dx = D2.min(axis=0)
    Dy = D2.min(axis=1)
    Qx = np.quantile(Dx,q)
    Qy = np.quantile(Dy,q)
    Q = np.max((Qx, Qy), axis=0)
    return np.sqrt(Q)
