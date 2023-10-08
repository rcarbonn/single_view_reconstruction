import numpy as np

def project_pts(P, X):
    X = np.hstack((X, np.ones([X.shape[0],1])))
    x = P@X.T
    x = x/x[-1,:]
    return x.T

def solve_dlt(x, X):
    n,_ = x.shape
    x = np.expand_dims(x,axis=1)
    cross_ = np.cross(x, np.identity(3)*-1)
    a = []
    for i in range(n):
        r = np.zeros((3,12))
        r[0,:4] = X[i]
        r[1,4:8] = X[i]
        r[2,8:] = X[i]
        p = cross_[i] @ r
        a.append(p[0])
        a.append(p[1])
    A = np.array(a)
    _,_,vh = np.linalg.svd(A)
    p = vh[-1]
    P = p.reshape(3,4)
    return P/P[-1,-1]


def computeP(pts2d, pts3d):

    assert pts2d.shape[0] == pts3d.shape[0], "2dpts and 3dpts should have equal number of points"
    n_pts,_ = pts2d.shape
    pts2d_ = np.hstack((pts2d, np.ones([n_pts,1])))
    pts3d_ = np.hstack((pts3d, np.ones([n_pts,1])))

    mean2d = np.mean(pts2d, axis=0)
    mean3d = np.mean(pts3d, axis=0)

    x = pts2d - mean2d
    X = pts3d - mean3d
    s2d = np.sqrt(2)/np.max(np.linalg.norm(x, axis=1))
    s3d = np.sqrt(3)/np.max(np.linalg.norm(X, axis=1))

    T2d = np.array([[s2d, 0, -s2d*mean2d[0]],
                    [0, s2d, -s2d*mean2d[1]],
                    [0, 0, 1]])
    T3d = np.array([[s3d, 0, 0, -s3d*mean3d[0]],
                    [0, s3d, 0, -s3d*mean3d[1]],
                    [0, 0, s3d, -s3d*mean3d[2]],
                    [0, 0, 0, 1]])
    xnorm = T2d@pts2d_.T
    Xnorm = T3d@pts3d_.T

    Pbar = solve_dlt(xnorm.T, Xnorm.T)
    P = np.linalg.inv(T2d)@Pbar@T3d
    return P/P[-1,-1]
