import numpy as np



# State preparation
up = np.array(
    [1, 0],
    dtype=complex
)
down = np.array(
    [0, 1],
    dtype=complex
)

state = (
    np.kron(np.kron(up, up), up) + np.kron(np.kron(down, down), down)
) / np.sqrt(2)

C = state.reshape(2, 2, 2)

print("State:", state)
print("Coefficient tensor shape:", C.shape)
print()



# Separate spin 1 from spins 2 and 3
M = C.reshape(2, 4)

U, s, Vh = np.linalg.svd(M, full_matrices=False)

A1 = U
remainder = np.diag(s) @ Vh

print("First tensor shape:", A1.shape)       # (2, 2)
print("Remainder shape:", remainder.shape)   # (2, 4)
print("Schmidt coefficients:", s)
print()



# Separate spins 2 and 3
bond1 = A1.shape[1]

M2 = remainder.reshape(bond1 * 2, 2)
U2, s2, Vh2 = np.linalg.svd(M2, full_matrices=False)

bond2 = len(s2)

A2 = U2.reshape(bond1, 2, bond2)
A3 = np.diag(s2) @ Vh2

print("A1 shape:", A1.shape)  # (2, 2):    i, alpha
print("A2 shape:", A2.shape)  # (2, 2, 2): alpha, j, beta
print("A3 shape:", A3.shape)  # (2, 2):    beta, k
