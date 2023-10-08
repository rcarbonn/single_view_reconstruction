import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def random_pattern():
    # Generate random pattern for 3 faces of the cube
    pts_right = np.random.uniform(0.1, 0.9, (1000,3))
    pts_right[:,0] = 1

    pts_left = np.random.uniform(0.1, 0.9, (1000,3))
    pts_left[:,1] = 0

    pts_top = np.random.uniform(0.1, 0.9, (1000,3))
    pts_top[:,2] = 1

    pts = np.vstack((pts_right, pts_left, pts_top))

    np.save('data/q1/cube_pts.npy', pts)


def extract_coords(impath):
    image = Image.open(impath)
    img = np.array(image)
    if len(img.shape) == 3:
        img = img[:,:,0]
    img[img!=255] = 0
    pts_mask = np.asarray(np.column_stack(np.where(img == 0)), dtype=np.float32)
    h,w = img.shape
    pts_mask[:,0] = pts_mask[:,0]/h
    pts_mask[:,1] = pts_mask[:,1]/w
    pts_mask = 1 - np.flip(pts_mask, axis=1)
    return pts_mask

def two_face_pts(impath1, impath2):

    pts1 = extract_coords(impath1)
    pts_right = np.ones((pts1.shape[0],3))
    pts_right[:,1:] = pts1

    pts2 = extract_coords(impath2)
    pts_left = np.zeros((pts2.shape[0],3))
    pts_left[:,0] = pts2[:,0]
    pts_left[:,2] = pts2[:,1]

    pts = np.vstack((pts_right, pts_left))
    np.save('data/q1/cube_pts3.npy', pts)

if __name__ == '__main__':
    # random_pattern()
    two_face_pts('data/tower1.jpg', 'data/colosseum1.jpg')
