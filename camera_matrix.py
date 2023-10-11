import numpy as np
from scipy.linalg import cholesky

from perspective import gen_lines_and_intersection

np.set_printoptions(suppress=True, precision=4)

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


def computeK(pts):
    # assuming zero skew and square pixels
    # pts of shape (3,2,2,3)
    _,_,vpt1 = gen_lines_and_intersection(pts[0][0], pts[0][1])
    _,_,vpt2 = gen_lines_and_intersection(pts[1][0], pts[1][1])
    _,_,vpt3 = gen_lines_and_intersection(pts[2][0], pts[2][1])

    pts_ = np.vstack((vpt1, vpt2, vpt3))
    # pts = np.vstack((vpt1[:2], vpt2[:2], vpt3[:2]))
    # m = np.mean(pts, axis=0)
    # pts = pts - m
    # s = np.sqrt(2)/np.max(np.linalg.norm(pts, axis=1))
    # T = np.array([[s, 0, -s*m[0]],
                  # [0, s, -s*m[1]],
                  # [0, 0, 1]])
    # pts = T@pts_.T
    # vpt1 = pts[:,0]
    # vpt2 = pts[:,1]
    # vpt3 = pts[:,2]

    # all 3 vanishing pts are orthogonal to each other
    A = np.zeros((3,4))
    A[0] = [vpt1[0]*vpt2[0]+vpt1[1]*vpt2[1], vpt1[0]*vpt2[2]+vpt1[2]*vpt2[0], vpt1[1]*vpt2[2]+vpt1[2]*vpt2[1], vpt1[2]*vpt2[2]]
    A[1] = [vpt2[0]*vpt3[0]+vpt2[1]*vpt3[1], vpt2[0]*vpt3[2]+vpt2[2]*vpt3[0], vpt2[1]*vpt3[2]+vpt2[2]*vpt3[1], vpt2[2]*vpt3[2]]
    A[2] = [vpt1[0]*vpt3[0]+vpt1[1]*vpt3[1], vpt1[0]*vpt3[2]+vpt1[2]*vpt3[0], vpt1[1]*vpt3[2]+vpt1[2]*vpt3[1], vpt1[2]*vpt3[2]]
    _,D,vh = np.linalg.svd(A)
    w = vh[-1]
    w = w/w[-1]
    W = np.zeros((3,3))
    W[0,0] = w[0]
    W[0,2] = w[1]
    W[1,1] = w[0]
    W[1,2] = w[2]
    W[2,0] = w[1]
    W[2,1] = w[2]
    W[2,2] = w[3]
    # conic = T.T @ W @ T
    conic = W
    U = cholesky(conic)
    K = np.linalg.inv(U)
    K = K/K[-1,-1]
    print("camera intrinsics:a \n", K)
    return K, pts_


def computeK2(h1s, h2s):
    n_constraints, _ = h1s.shape

    # pts1 = h1s[:,:2]
    # pts2 = h2s[:,:2]
    # m1 = np.mean(pts1, axis=0)
    # m2 = np.mean(pts2, axis=0)
    # pts1 = pts1 - m1
    # pts2 = pts2 - m2
    # s1 = np.sqrt(1)/np.max(np.linalg.norm(pts1, axis=1))
    # s2 = np.sqrt(1)/np.max(np.linalg.norm(pts2, axis=1))
    # T1 = np.array([[s1, 0, -s1*m1[0]],
    #                [0, s1, -s1*m1[1]],
    #                [0, 0, 1]])
    # T2 = np.array([[s2, 0, -s2*m2[0]],
    #                [0, s2, -s2*m2[1]],
    #                [0, 0, 1]])
    # pts1 = T1@h1s.T
    # pts2 = T2@h2s.T
    # h1s = pts1.T
    # h2s = pts2.T

    A = np.zeros((2*n_constraints, 6))
    i=0
    for h1,h2 in zip(h1s, h2s):
        A[i] = [h1[0]*h2[0], h1[0]*h2[1]+h1[1]*h2[0], h1[0]*h2[2]+h1[2]*h2[0], h1[1]*h2[1], h1[1]*h2[2]+h1[2]*h2[1], h1[2]*h2[2]]
        A[i+1] = [h1[0]*h1[0]-h2[0]*h2[0], 2*(h1[0]*h1[1]-h2[0]*h2[1]), 2*(h1[0]*h1[2]-h2[0]*h2[2]), h1[1]*h1[1]-h2[1]*h2[1], 2*(h1[1]*h1[2]-h2[1]*h2[2]), h1[2]*h1[2]-h2[2]*h2[2]]
        i+=2
    _,D,vh = np.linalg.svd(A)
    w = vh[-1]
    w = w/w[-1]
    W = np.zeros((3,3))
    W[0,0] = w[0]
    W[0,1] = w[1]
    W[0,2] = w[2]
    W[1,0] = w[1]
    W[1,1] = w[3]
    W[1,2] = w[4]
    W[2,0] = w[2]
    W[2,1] = w[4]
    W[2,2] = w[5]
    # conic = T1.T @ W @ T2
    conic = W
    U = cholesky(conic)
    K = np.linalg.inv(U)
    K = K/K[-1,-1]
    print("camera intrinsics:b \n", K)
    return K, conic
