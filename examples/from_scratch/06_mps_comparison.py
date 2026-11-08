import numpy as np



def state_to_mps(state, n_spins, tol=1e-12):
    tensors = []
    remainder = state.reshape(1, -1)
    left_bond = 1

    for site in range(n_spins - 1):
        matrix = remainder.reshape(left_bond * 2, -1)
        U, s, Vh = np.linalg.svd(matrix, full_matrices=False)

        right_bond = max(1, int(np.count_nonzero(s > tol)))

        U = U[:, :right_bond]
        s = s[:right_bond]
        Vh = Vh[:right_bond, :]

        A = U.reshape(left_bond, 2, right_bond)
        tensors.append(A)

        remainder = np.diag(s) @ Vh
        left_bond = right_bond
    
    tensors.append(remainder.reshape(left_bond, 2, 1))

    return tensors


def mps_to_state(tensors):
    result = tensors[0]

    for A in tensors[1:]:
        result = np.tensordot(result, A, axes=([-1], [0]))
    
    return result.reshape(-1)



rng = np.random.default_rng(42)

for n_spins in [2, 4, 8, 12]:
    dimension = 2**n_spins

    # Dense product state: |+> tensor ... tensor |+>
    product = np.ones(dimension, dtype=complex) / np.sqrt(dimension)

    # GHZ state
    ghz = np.zeros(dimension, dtype=complex)
    ghz[0] = ghz[-1] = 1 / np.sqrt(2)

    # Random normalized pure state
    random_state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    random_state /= np.linalg.norm(random_state)

    print(f"\nSpins: {n_spins}")

    for name, state in [
        ("Product", product),
        ("GHZ", ghz),
        ("Random", random_state),
    ]:
        tensors = state_to_mps(state, n_spins)

        bonds = [A.shape[2] for A in tensors[:-1]]
        mps_entries = sum(A.size for A in tensors)

        print(f"\n  {name}")
        print("    Bond dimensions:", bonds)
        print("    Full-state entries:", state.size)
        print("    MPS entries:", mps_entries)
