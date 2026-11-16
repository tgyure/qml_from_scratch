import numpy as np


def state_to_mps(state, n_spins, tol=1e-12):
    state = np.asarray(state, dtype=complex)

    tensors = []
    remainder = state.copy().reshape(1, -1)
    left_bond = 1

    for site in range(n_spins - 1):
        matrix = remainder.reshape(left_bond * 2, -1)
        U, s, Vh = np.linalg.svd(matrix, full_matrices=False)

        right_bond = max(1, int(np.count_nonzero(s > tol)))

        tensors.append(
            U[:, :right_bond].reshape(left_bond, 2, right_bond)
        )

        remainder = np.diag(s[:right_bond]) @ Vh[:right_bond, :]
        left_bond = right_bond

    tensors.append(remainder.reshape(left_bond, 2, 1))

    return tensors


def mps_to_state(tensors):
    result = tensors[0]

    for A in tensors[1:]:
        result = np.tensordot(result, A, axes=([-1], [0]))

    return result.reshape(-1).copy()


def bond_dimensions(tensors):
    return [A.shape[2] for A in tensors[:-1]]


def stored_entries(tensors):
    return sum(A.size for A in tensors)
