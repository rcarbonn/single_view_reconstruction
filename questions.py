import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from camera_matrix import computeP, project_pts, computeK, computeK2
from utils import add_lines, annotate, annotate_parallel, reshape_plane_annots
from perspective import vanish_shift, plane_angles, plane_normal
from homography import get_homography

DATA_CONFIG = {

    'q1': {
        'img': 'data/q1/bunny.jpeg',
        'img_cube': 'data/q1/cube1.png',
        'corr': 'data/q1/bunny.txt',
        'corr_cube': 'data/q1/cube1.txt',
        'bd': 'data/q1/bunny_bd.npy',
        'pts': 'data/q1/bunny_pts.npy',
        'cube_pts': 'data/q1/cube_pts3.npy',
        },
    'q2': {
        'img1': 'data/q2a.png',
        'img2': 'data/q2b.png',
        'annot1': 'data/q2/q2a.npy',
        'annot2': 'data/q2/q2b.npy',
        },
    'q3': {
        'img': 'data/q3.png',
        'annot': 'data/q3/q3.npy',
        }

}

def q1(args):
    img = DATA_CONFIG['q1']['img']
    corr = DATA_CONFIG['q1']['corr']
    bd = DATA_CONFIG['q1']['bd']
    pts = DATA_CONFIG['q1']['pts']
    image = Image.open(img)
    corr = np.loadtxt(corr)
    pts2d = corr[:,:2]
    pts3d = corr[:,2:]
    P = computeP(pts2d, pts3d)
    pts = np.load(pts, allow_pickle=True)
    bd = np.load(bd, allow_pickle=True)
    bd = bd.flatten().reshape(-1,3)
    img_pts = project_pts(P, pts)
    lines = project_pts(P, bd)

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.scatter(img_pts[:,0], img_pts[:,1], c='r', s=1)
    add_lines(ax, lines, ptype=args.viz)

    # q1b
    if os.path.isfile(DATA_CONFIG['q1']['corr_cube']):
        img_cube = DATA_CONFIG['q1']['img_cube']
        corr_cube = DATA_CONFIG['q1']['corr_cube']
        image_cube = Image.open(img_cube)
        corr_cube = np.loadtxt(corr_cube)
        pts2d_cube = corr_cube[:,:2]
        pts3d_cube = corr_cube[:,2:]
        P_cube = computeP(pts2d_cube, pts3d_cube)
        pts_cube = np.load(DATA_CONFIG['q1']['cube_pts'], allow_pickle=True)
        img_pts = project_pts(P_cube, pts_cube)
        img_pts = img_pts[::2]
        fig, ax = plt.subplots()
        ax.imshow(image_cube)
        ax.scatter(img_pts[:,0], img_pts[:,1], c='r', s=0.01)
        plt.show()
    else:
        img_cube = DATA_CONFIG['q1']['img_cube']
        pts = annotate(img_cube)
        np.savetxt(DATA_CONFIG['q1']['corr_cube'], pts)


def q2(args):

    img1 = DATA_CONFIG['q2']['img1']
    if not args.new_annot:
        annotations1 = DATA_CONFIG['q2']['annot1']
        annot1 = np.load(annotations1, allow_pickle=True) # shape (3,2,4)
    else:
        print('New annotations')
        annot1 = annotate_parallel(DATA_CONFIG['q2']['img1'])
        np.save('data/q2/q2a_custom2.npy', annot1)
    n_annots, _, _ = annot1.shape
    annots1 = annot1.reshape(-1,2)
    annots1 = np.hstack((annots1, np.ones([annots1.shape[0],1])))
    pts = annots1.reshape(n_annots, 2, 2, 3)
    K, vpts = computeK(pts)
    result, Ht = vanish_shift(np.array(Image.open(img1)), vpts[:,:2])
    line_pts = (Ht@vpts.T).T
    lines = np.array([line_pts[0], line_pts[1], line_pts[1], line_pts[2], line_pts[0], line_pts[2]])
    ppoint = (Ht@K[:,2]).T
    fig, ax = plt.subplots()
    ax.imshow(result)
    add_lines(ax, lines, ptype='lines')
    plot_points = np.vstack((line_pts, ppoint))
    ax.scatter(plot_points[:,0], plot_points[:,1], c='g', s=50)
    plt.show()

    # q2b
    img2 = DATA_CONFIG['q2']['img2']
    annot2 = np.load(DATA_CONFIG['q2']['annot2'], allow_pickle=True)
    # src_pts = np.array([[0,0],[1,0],[1,1],[0,1]])
    src_pts = np.array([[1,0],[1,1],[0,1],[0,0]])
    h1s = []
    h2s = []
    for dpts in annot2:
        H = get_homography(src_pts, dpts)
        h1s.append(H[:,0])
        h2s.append(H[:,1])
    K, conic = computeK2(np.array(h1s), np.array(h2s))
    plane_angles(annot2[0], annot2[1], conic)
    plane_angles(annot2[1], annot2[2], conic)
    plane_angles(annot2[0], annot2[2], conic)

def q3(args):
    image = DATA_CONFIG['q3']['img']
    img = np.array(Image.open(image))
    annotations = DATA_CONFIG['q3']['annot']
    annots = np.load(annotations, allow_pickle=True)
    n_annots, _, _ = annots.shape
    annots_all = annots.reshape(-1,2)
    annots_all_ = np.hstack((annots_all, np.ones([annots_all.shape[0],1])))
    annot_ids = [0,1,3,2,4,7,5,6,11,8,10,9]
    annots123 = annots_all[annot_ids,:].reshape(-1,2,4)
    pts = reshape_plane_annots(annots123)
    K, vpts = computeK(pts)
    Kinv = np.linalg.inv(K)
    normals = np.array([Kinv @ plane_normal(annots[i]).reshape(-1,1) for i in range(n_annots)])
    directions = (Kinv @ annots_all_.T).T
    # set depth of point 2 = 20 mts along the camera
    pt2 = 20*directions[1].reshape(-1,1)
    # point 2 lies on plane 1,2,4,5
    dp1 = -normals[0].T@pt2
    dp2 = -normals[1].T@pt2
    dp4 = -normals[3].T@pt2
    dp5 = -normals[4].T@pt2
    # point 3 lies on plane 1 and 3
    l3 = -dp1/directions[2].dot(normals[0])
    pt3 = l3*directions[2].reshape(-1,1)
    dp3 = -normals[2].T@pt3
    plane_depths = [dp1, dp2, dp3, dp4, dp5]
    h,w,c = img.shape
    pts3d = []
    for i in range(n_annots):
    # for i in range(1):
        mask = cv2.fillConvexPoly(np.zeros((h,w)), annots[i].astype(np.int32), 1)
        pts = np.asarray(np.column_stack(np.where(mask == 1)), dtype=np.float32)
        pts = np.flip(pts, axis=1)
        pts_ = np.hstack((pts, np.ones([pts.shape[0],1])))
        pdirs = (Kinv @ pts_.T).T
        l = -plane_depths[i]/pdirs.dot(normals[i])
        plane_pts = l*pdirs 
        pts3d.append(plane_pts)
        # plt.imshow(mask)
        # plt.show()
    pts3d = np.vstack(pts3d)
    pts3d = pts3d[::10]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2], c='r', s=0.1)
    plt.show()
