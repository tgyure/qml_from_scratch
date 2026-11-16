import numpy as np

from .canonical import canonicalize


def one_site_gate(tensors, gate, site):
    gate = np.asarray(gate, dtype=complex)
    physical = tensors[site].shape[1]

    tensors[site] = np.einsum("ij,ajb->aib", gate, tensors[site])


def two_site_gate(tensors, gate, site):
    physical_left = tensors[site].shape[1]
    physical_right = tensors[site + 1].shape[1]
    dimension = physical_left * physical_right

    gate = np.asarray(gate, dtype=complex)

    # Make the surrounding block states orthonormal.
    canonicalize(tensors, center=site)

    # Join the neighboring tensors.
    theta = np.tensordot(tensors[site], tensors[site + 1], axes=([-1], [0]))

    # Apply the gate to their physical indices.
    gate = gate.reshape(
        physical_left, physical_right,
        physical_left, physical_right
    )
    theta = np.einsum("ijpq,apqc->aijc", gate, theta)

    # Split the updated pair back into two MPS tensors.
    left_bond = theta.shape[0]
    right_bond = theta.shape[-1]
    matrix = theta.reshape(
        left_bond * physical_left,
        physical_right * right_bond
    )

    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
    bond = len(s)

    tensors[site] = U.reshape(left_bond, physical_left, bond)
    tensors[site + 1] = (np.diag(s) @ Vh).reshape(bond, physical_right, right_bond)
