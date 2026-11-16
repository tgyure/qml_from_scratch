import numpy as np


def mps_overlap(tensors_a, tensors_b):
    environment = np.ones((1, 1), dtype=complex)
    
    for A, B in zip(tensors_a, tensors_b):
        environment = np.einsum(
            "ac,aib,cid->bd",
            environment,
            A.conj(),
            B,
            optimize=True,
        )

    return environment[0, 0]


def mps_matrix_element(tensors, operators=None):
    if operators is None:
        operators = {}

    environment = np.ones((1, 1), dtype=complex)

    for site, A in enumerate(tensors):
        physical = A.shape[1]
        operator = np.asarray(
            operators.get(site, np.eye(physical)),
            dtype=complex,
        )

        environment = np.einsum(
            "ac,aib,ij,cjd->bd",
            environment,
            A.conj(),
            operator,
            A,
            optimize=True,
        )

    return environment[0, 0]


def mps_expectation(tensors, operators=None):
    norm_squared = mps_overlap(tensors, tensors).real

    return mps_matrix_element(tensors, operators) / norm_squared


def mps_norm(tensors):
    norm_squared = mps_overlap(tensors, tensors).real
    return float(np.sqrt(max(0.0, norm_squared)))


def mps_fidelity(tensors_a, tensors_b):
    norm_a_squared = mps_overlap(tensors_a, tensors_a).real
    norm_b_squared = mps_overlap(tensors_b, tensors_b).real

    overlap = mps_overlap(tensors_a, tensors_b)

    fidelity = abs(overlap)**2 / (norm_a_squared * norm_b_squared)

    return float(np.clip(fidelity, 0.0, 1.0))
