import numpy as np
import cv2

from utils import split_annotations

def proj_line(pt1, pt2):
    l = np.cross(pt1, pt2)
    return l/l[-1]

def intersection_pt(l1, l2):
    pt = np.cross(l1, l2)
    return pt/pt[-1]

def gen_lines_and_intersection(pts1, pts2):
    # get a line between pts1 and another line between pts2 and intersection among them
    l1 = proj_line(pts1[0], pts1[1])
    l2 = proj_line(pts2[0], pts2[1])
    p = intersection_pt(l1, l2)
    return l1,l2,p

def angle_change(l1, l2, Hline):
    l1_ = (Hline @ l1.reshape(-1,1)).flatten()
    l2_ = (Hline @ l2.reshape(-1,1)).flatten()
    l1 = l1[:2]
    l2 = l2[:2]
    l1_ = (l1_/l1_[-1])[:2]
    l2_ = (l2_/l2_[-1])[:2]
    angle_before = np.dot(l1, l2) / (np.linalg.norm(l1) * np.linalg.norm(l2))
    angle_after = np.dot(l1_, l2_) / (np.linalg.norm(l1_) * np.linalg.norm(l2_))
    return angle_before, angle_after

def rectify_annots(img, annots, H):
    Ht = perspective_shift(img, H)
    annotsh = np.hstack((annots, np.ones(annots.shape[0]).reshape(-1,1)))
    annots = annots.reshape(-1,1,2)
    rectified_annots = cv2.perspectiveTransform(annots, Ht@H)
    ra = (H@annotsh.T).T
    ra = ra/ra[:,None,-1]
    # print(ra[:,:2])
    return rectified_annots.squeeze(1)

def perspective_shift(img, H):
    h, w = img.shape[:2]
    pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float64).reshape(-1, 1, 2)
    pts = cv2.perspectiveTransform(pts, H)
    [xmin, ymin] = (pts.min(axis=0).ravel() - 0.5).astype(int)
    [xmax, ymax] = (pts.max(axis=0).ravel() + 0.5).astype(int)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    return Ht

def vanish_shift(img, vpts):
    h,w = img.shape[:2]
    pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float64)
    pts = np.vstack((pts, vpts))
    [xmin, ymin] = (pts.min(axis=0).ravel() - 250.5).astype(int)
    [xmax, ymax] = (pts.max(axis=0).ravel() + 250.5).astype(int)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float64)
    result = np.ones((ymax-ymin, xmax-xmin, 3), dtype=np.uint8)*255
    cv2.warpPerspective(img, Ht, (xmax-xmin, ymax-ymin), result, borderValue=(255,255,255))
    return result, Ht

def plane_normal(pts, Kinv):
    pts_ = np.zeros_like(pts)
    pts_[0] = pts[0]
    pts_[1] = pts[3]
    pts_[2] = pts[1]
    pts_[3] = pts[2]

    pts, _ = split_annotations(pts)
    pts_, _ = split_annotations(pts_)
    _,_,vp1 = gen_lines_and_intersection(pts[0], pts[1])
    _,_,vp2 = gen_lines_and_intersection(pts_[0], pts_[1])
    # vl = proj_line(vp1, vp2)
    vl = np.cross(Kinv @ vp1, Kinv @vp2)
    return vl

def plane_angles(pts1, pts2, conic):

    pts1_ = np.zeros_like(pts1)
    pts1_[0] = pts1[0]
    pts1_[1] = pts1[3]
    pts1_[2] = pts1[1]
    pts1_[3] = pts1[2]

    pts1, _ = split_annotations(pts1)
    pts1_, _ = split_annotations(pts1_)
    _,_,vp1 = gen_lines_and_intersection(pts1[0], pts1[1])
    _,_,vp2 = gen_lines_and_intersection(pts1_[0], pts1_[1])
    vl1 = proj_line(vp1, vp2)

    pts2_ = np.zeros_like(pts2)
    pts2_[0] = pts2[0]
    pts2_[1] = pts2[3]
    pts2_[2] = pts2[1]
    pts2_[3] = pts2[2]
    pts2, _ = split_annotations(pts2)
    pts2_, _ = split_annotations(pts2_)
    _,_,vp1 = gen_lines_and_intersection(pts2[0], pts2[1])
    _,_,vp2 = gen_lines_and_intersection(pts2_[0], pts2_[1])
    vl2 = proj_line(vp1, vp2)

    wi = np.linalg.inv(conic)
    vl1_ = vl1.T @ wi @ vl1
    vl2_ = vl2.T @ wi @ vl2
    ct = (vl1.T @ wi @ vl2) / (np.sqrt(vl1_) * np.sqrt(vl2_))
    print(np.arccos(ct)*180/np.pi)
