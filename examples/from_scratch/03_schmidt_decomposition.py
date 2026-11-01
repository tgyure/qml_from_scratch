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

plus = (up + down) / np.sqrt(2)

product_state = np.kron(plus, up)
bell_state = (
    np.kron(up, up) + np.kron(down, down)
) / np.sqrt(2)

C_product = product_state.reshape(2, 2)
C_bell = bell_state.reshape(2, 2)



# State reconstruction via Schmidt decomposition
for name, C in [("Product", C_product), ("Bell", C_bell)]:
    U, s, Vh = np.linalg.svd(C, full_matrices=False)

    reconstructed_state = np.zeros(C.size, dtype=complex)

    print(f"\n{name}")
    print("Singular values:", s)

    for alpha, weight in enumerate(s):
        a = U[:, alpha]
        b = Vh[alpha, :]

        term = weight * np.kron(a, b)
        reconstructed_state += term

        print(f"\nTerm {alpha}:")
        print("  Weight:", weight)
        print("  First spin:", a)
        print("  Second spin:", b)

    # Density matrices and entanglement entropy
    rho_A = C @ C.conj().T
    rho_B = C.T @ C.conj()

    print("\nrho_A:\n", rho_A)
    print("rho_B:\n", rho_B)

    for label, rho in [("A", rho_A), ("B", rho_B)]:
        eigenvalues = np.linalg.eigvalsh(rho)

        # Ignore tiny numerical errors near zero.
        probabilities = eigenvalues[eigenvalues > 1e-14]
        entropy = -np.sum(probabilities * np.log2(probabilities))

        print(f"\nSubsystem {label}:")
        print("  Eigenvalues:", eigenvalues)
        print("  Entropy:", entropy)
