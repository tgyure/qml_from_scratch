import numpy as np



def mps_CNOT(tensors, CNOT, site_i, site_j):
    theta = np.tensordot(tensors[site_i], tensors[site_j], axes=([-1], [0]))
    gate = CNOT.reshape(2, 2, 2, 2)

    theta = np.einsum("ijpq,apqc->aijc", gate, theta)

    left_bond = theta.shape[0]
    right_bond = theta.shape[-1]
    matrix = theta.reshape(left_bond * 2, 2 * right_bond)

    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
    bond = max(1, int(np.count_nonzero(s > 1e-12)))

    tensors[site_i] = U[:, :bond].reshape(left_bond, 2, bond)
    tensors[site_j] = (np.diag(s[:bond]) @ Vh[:bond, :]).reshape(
        bond, 2, right_bond
    )

    print(
        f"Tensor shapes after CNOT ({site_i + 1}{site_j + 1}):",
        [A.shape for A in tensors]
    )




# Setup
n_spins = 4

up = np.array(
    [1, 0],
    dtype=complex
)

tensors = [up.reshape(1, 2, 1).copy() for i in range(n_spins)]

print("Initial tensor shapes:        ", [A.shape for A in tensors])



# Hadamard acts on A1.
H = np.array(
    [[1, 1],
    [1, -1]]
) / np.sqrt(2)

tensors[0] = np.einsum("ij,ajb->aib", H, tensors[0])

print("Tensor shapes after Hadamard: ", [A.shape for A in tensors])



# CNOT entangles sites 1 and 2, then 2 and 3, and finally 3 and 4.
CNOT = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]],
     dtype=complex
)

mps_CNOT(tensors, CNOT, 0, 1)
mps_CNOT(tensors, CNOT, 1, 2)
mps_CNOT(tensors, CNOT, 2, 3)
