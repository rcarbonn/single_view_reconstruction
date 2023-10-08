import os
import numpy as np
from PIL import Image
import argparse as ap
import matplotlib.pyplot as plt

from camera_matrix import computeP, project_pts
from utils import add_lines, annotate

DATA_CONFIG = {
        'q1': {
            'img': 'data/q1/bunny.jpeg',
            'img_cube': 'data/q1/cube1.png',
            'corr': 'data/q1/bunny.txt',
            'corr_cube': 'data/q1/cube1.txt',
            'bd': 'data/q1/bunny_bd.npy',
            'pts': 'data/q1/bunny_pts.npy',
            'cube_pts': 'data/q1/cube_pts3.npy',
            }
        }


def main(args):

    if args.question == 'q1':
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


    pass


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-q', '--question', choices=['q1', 'q2', 'q3', 'q4', 'q5'], required=True)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-i', '--image', default=None)
    parser.add_argument('-v', '--viz', default='lines')
    args = parser.parse_args()
    main(args)
