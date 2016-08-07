import numpy as np
import PIL.Image

def gen_grid(lim, width):
    ls = np.tile(np.linspace(-lim, lim, width), (width, 1)).astype('float32')
    return np.concatenate((ls.reshape(width*width, 1), \
            ls.T.reshape(width*width, 1)), axis=1)

from itertools import product
def mat_to_tileimg(X, img_shape, tile_shape):
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
    return PIL.Image.fromarray(fig)

def img_to_tileimg(X, tile_shape):
    assert(np.prod(tile_shape) >= len(X))
    assert(X.ndim == 4)
    img_shape = X.shape[2:]
    fig = np.zeros((img_shape[0]*tile_shape[0], img_shape[1]*tile_shape[1]),
            dtype='uint8')
    for (i, j) in product(range(tile_shape[0]), range(tile_shape[1])):
        ind = i*tile_shape[1] + j
        if ind < len(X):
            fig[i*img_shape[0]:(i+1)*img_shape[0],
                j*img_shape[1]:(j+1)*img_shape[1]] = 255*X[ind][0]
    return PIL.Image.fromarray(fig)
