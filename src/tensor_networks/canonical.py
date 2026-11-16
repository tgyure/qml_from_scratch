import numpy as np

from .contractions import mps_norm


def normalize_mps(tensors):
    norm = mps_norm(tensors)

    tensors[-1] = tensors[-1] / norm

    return norm


def canonicalize(tensors, center):
    # Sweep from the left edge toward center.
    for site in range(center):
        left_bond, physical, right_bond = tensors[site].shape
        matrix = tensors[site].reshape(left_bond * physical, right_bond)

        U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
        bond = len(s)

        tensors[site] = U.reshape(left_bond, physical, bond)

        remainder = np.diag(s) @ Vh
        tensors[site + 1] = np.tensordot(remainder, tensors[site + 1], axes=([1], [0]))

    # Sweep from the right edge toward center.
    for site in range(len(tensors) - 1, center, -1):
        left_bond, physical, right_bond = tensors[site].shape
        matrix = tensors[site].reshape(left_bond, physical * right_bond)

        U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
        bond = len(s)

        tensors[site] = Vh.reshape(bond, physical, right_bond)

        remainder = U @ np.diag(s)
        tensors[site - 1] = np.tensordot(tensors[site - 1], remainder, axes=([-1], [0]))


def truncate_bond(tensors, site, max_bond, tol=1e-12):
    canonicalize(tensors, center=site)

    theta = np.tensordot(tensors[site], tensors[site + 1], axes=([-1], [0]))

    left_bond = theta.shape[0]
    right_bond = theta.shape[-1]
    matrix = theta.reshape(left_bond * 2, 2 * right_bond)

    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)

    norm = np.linalg.norm(s)

    bond = max(1, int(np.count_nonzero(s > tol)))
    bond = min(bond, max_bond)

    discarded_weight = float(np.sum((s[bond:] / norm)**2))

    kept_s = s[:bond] / np.linalg.norm(s[:bond])

    tensors[site] = U[:, :bond].reshape(left_bond, 2, bond)
    tensors[site + 1] = (np.diag(kept_s) @ Vh[:bond, :]).reshape(bond, 2, right_bond)

    return discarded_weight
