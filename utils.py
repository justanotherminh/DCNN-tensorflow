import os
import cPickle as pickle
import numpy as np


def get_data_batch(data, label, batch_size):
    N = data.shape[0]
    mask = np.random.choice(N, batch_size, replace=False)
    return data[mask, :], label[mask, :]


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT, load_test=True):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 2):  # Change to (1, 6) for entire dataset
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    m = Xtr.mean(axis=0)
    s = Xtr.std(axis=0)
    Xtr -= m
    Xtr /= s
    ytr = np.zeros([Ytr.size, 10])
    ytr[np.arange(Ytr.size), Ytr] = 1
    del Ytr
    if load_test:
        Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        Xte -= m
        Xte /= s
        yte = np.zeros([Yte.size, 10])
        yte[np.arange(Ytr.size), Ytr] = 1
        del Yte
        return Xtr, ytr, Xte, yte
    else:
        return Xtr, ytr
