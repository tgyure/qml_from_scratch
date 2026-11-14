import numpy as np



up = np.array(
    [1, 0],
    dtype=complex
)
down = np.array(
    [0, 1],
    dtype=complex
)

state = np.kron(up, up)



H = np.array(
    [[1, 1],
    [1, -1]],
    dtype=complex
) / np.sqrt(2)

CNOT = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]],
     dtype=complex
)

identity = np.eye(2, dtype=complex)



state_after_H = np.kron(H, identity) @ state
state_after_CNOT = CNOT @ state_after_H

print("Initial state:   ", state)
print("After Hadamard:  ", state_after_H)
print("After CNOT:      ", state_after_CNOT)
print()



A1 = up.reshape(1, 2, 1).copy()
A2 = up.reshape(1, 2, 1).copy()

A1 = np.einsum("ij,ajb->aib", H, A1)

# Reconstruct only to check against the full-vector result.
mps_after_H = np.tensordot(A1, A2, axes=([-1], [0])).reshape(-1)

print("MPS after Hadamard:", mps_after_H)
print("Difference:", np.linalg.norm(mps_after_H - state_after_H))
print()



theta = np.tensordot(A1, A2, axes=([-1], [0]))

gate = CNOT.reshape(2, 2, 2, 2)
theta = np.einsum("ijpq,apqc->aijc", gate, theta)

left_bond = theta.shape[0]
right_bond = theta.shape[-1]
matrix = theta.reshape(left_bond * 2, 2 * right_bond)



U, s, Vh = np.linalg.svd(matrix, full_matrices=False)

bond = max(1, int(np.count_nonzero(s > 1e-12)))

A1 = U[:, :bond].reshape(left_bond, 2, bond)
A2 = (np.diag(s[:bond]) @ Vh[:bond, :]).reshape(
    bond, 2, right_bond
)

mps_after_CNOT = np.tensordot(
    A1, A2, axes=([-1], [0])
).reshape(-1)

print("MPS after CNOT:", mps_after_CNOT)
print("Difference:", np.linalg.norm(mps_after_CNOT - state_after_CNOT))
print("Tensor shapes:", A1.shape, A2.shape)
print("Schmidt coefficients:", s[:bond])
