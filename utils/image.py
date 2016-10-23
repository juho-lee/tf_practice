import numpy as np
import PIL.Image

def gen_grid(lim, width):
    ls = np.tile(np.linspace(-lim, lim, width), (width, 1)).astype('float32')
    return np.concatenate((ls.reshape(width*width, 1), \
            ls.T.reshape(width*width, 1)), axis=1)

from itertools import product
def batchmat_to_tileimg(X, img_shape, tile_shape, border=True):
    assert(np.prod(tile_shape) >= len(X))
    assert(X.ndim == 2)
    fig = np.zeros((img_shape[0]*tile_shape[0], img_shape[1]*tile_shape[1]),
            dtype='uint8')
    for (i, j) in product(range(tile_shape[0]), range(tile_shape[1])):
        rowind = i*tile_shape[1] + j
        if rowind < len(X):
            cell = X[rowind].reshape(img_shape)
            fig[i*img_shape[0]:(i+1)*img_shape[0],
                j*img_shape[1]:(j+1)*img_shape[1]] = 255*cell
            if border:
                fig[(i+1)*img_shape[0]-1,j*img_shape[1]:(j+1)*img_shape[1]] = 255
                fig[i*img_shape[0]:(i+1)*img_shape[0],(j+1)*img_shape[1]-1] = 255

    return PIL.Image.fromarray(fig)

def batchimg_to_tileimg(X, tile_shape, channel_dim=3, border=True):
    assert(np.prod(tile_shape) >= len(X))
    assert(X.ndim == 4)
    assert(channel_dim==1 or channel_dim==3)
    if channel_dim==3:
        X = X.transpose([0,3,1,2])
    img_shape = X.shape[2:]
    fig = np.zeros((img_shape[0]*tile_shape[0], img_shape[1]*tile_shape[1]),
            dtype='uint8')
    for (i, j) in product(range(tile_shape[0]), range(tile_shape[1])):
        ind = i*tile_shape[1] + j
        if ind < len(X):
            fig[i*img_shape[0]:(i+1)*img_shape[0],
                j*img_shape[1]:(j+1)*img_shape[1]] = 255*X[ind][0]
            if border:
                fig[(i+1)*img_shape[0]-1,j*img_shape[1]:(j+1)*img_shape[1]] = 255
                fig[i*img_shape[0]:(i+1)*img_shape[0],(j+1)*img_shape[1]-1] = 255

    return PIL.Image.fromarray(fig)

import matplotlib.pyplot as plt
def create_fig(name, gray=True, axisoff=True):
    fig = plt.figure(name)
    if gray:
        plt.gray()
    if axisoff:
        plt.axis('off')
    return fig
        
