import numpy as np

up = np.array(
    [1, 0],
    dtype=complex
)
down = np.array(
    [0, 1],
    dtype=complex
)

plus = (up + down) / np.sqrt(2)

product_state = np.kron(plus, up)
bell_state = (
    np.kron(up, up) + np.kron(down, down)
) / np.sqrt(2)

C_product = product_state.reshape(2, 2)
C_bell = bell_state.reshape(2, 2)

print("Product coefficient matrix:\n", C_product)
print("Bell coefficient matrix:\n", C_bell)
print()

C_from_factors = np.outer(plus, up)

print("Constructed from individual states:\n", C_from_factors)
print()

print("Product rank:", np.linalg.matrix_rank(C_product))
print("Bell rank:   ", np.linalg.matrix_rank(C_bell))
