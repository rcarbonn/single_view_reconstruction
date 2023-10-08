import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D as l2d

def split_annotations(annots):
    n_annots,_ = annots.shape
    annots_ = np.hstack((annots, np.ones([n_annots,1], annots.dtype)))
    annots_ = np.split(annots_, n_annots//2, axis=0)
    annots = np.split(annots, n_annots//2, axis=0)
    return annots_, annots

def add_lines(ax, line_annots, ptype='lines', cols='random'):
    n,_ = line_annots.shape
    ldata = np.split(line_annots.T, n//2, axis=1)
    if cols=='random':
        colors = np.repeat(np.random.uniform(0, 1, (n//2,1,3)), 2, axis=0)
    else:
        colors=cols
    if ptype=='lines':
        for i,l in enumerate(ldata):
            line = l2d(l[0], l[1], color=colors[i], linewidth=1.0)
            ax.add_line(line)
    elif ptype=='scatter':
        ax.scatter(line_annots[:,0], line_annots[:,1])

def annotate(impath):
    im = Image.open(impath)
    im = np.array(im)

    clicks = []

    def click(event):
        x, y = event.xdata, event.ydata
        clicks.append([x, y, 1.])

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.imshow(im)

    _ = fig.canvas.mpl_connect('button_press_event', click)
    plt.show()

    return clicks

def annotate_parallel(impath):
    im = Image.open(impath)
    im = np.array(im)

    clicks = []

    def click(event):
        x, y = event.xdata, event.ydata
        clicks.append([x, y])

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.imshow(im)

    _ = fig.canvas.mpl_connect('button_press_event', click)
    plt.show()

    clicks = np.array(clicks).reshape(3,2,4)

    return clicks
