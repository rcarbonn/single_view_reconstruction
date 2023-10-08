import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from camera_matrix import computeP, project_pts, computeK
from utils import add_lines, annotate, annotate_parallel

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

    if not args.new_annot:
        img1 = DATA_CONFIG['q2']['img1']
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
    computeK(pts)

def q3(args):
    pass
