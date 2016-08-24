import gzip
import cPickle as pkl

def load_pkl(filename):
    f = gzip.open(filename, 'rb')
    data = pkl.load(f)
    f.close()
    return data
