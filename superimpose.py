# -*- coding: utf-8 -*-

import numpy as np


def superimpose(X, X_ref, weights):
    """
    Superimpose two sets of points.

    Parameters
    ----------
    X : np.ndarray, size = (length, 3)
        Cartesian coordinate for each point. A row means a point.
    X_ref : np.ndarray, size = (length, 3)
        Reference position of X.
    weights : np.array, size = (length,)
        Weights for each points.

    Returns
    -------
    X_fit : np.ndarray, size = (length, 3)
        Superimposed cartesian coordinates. Its center and directions
        are fitted to `X_ref`.
    """
    X_mean = np.average(X, axis=0, weights=weights)
    X_ref_mean = np.average(X_ref, axis=0, weights=weights)

    X_trans = X - X_mean
    X_ref_trans = X_ref - X_ref_mean

    corr_mat = np.dot(X_trans.T, X_ref_trans)
    u, s, v = np.linalg.svd(corr_mat)
    rot_mat = np.dot(v, u.T)

    return np.dot(rot_mat, X_trans.T).T + X_ref_mean


def example():
    X = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0],
    ])
    X_ref = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ])
    weights = np.array([1.0, 1.0, 1.0])

    X_fit = superimpose(X, X_ref, weights)

    print(X_fit)


if __name__ == '__main__':
    example()
