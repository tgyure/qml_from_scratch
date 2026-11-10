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


# Contract <psi| O_0 tensor ... tensor O_(N-1) |psi>.
# Operators are specified by site; unspecified sites use identity.
def mps_matrix_element(tensors, operators=None):
    if operators is None:
        operators = {}

    environment = np.ones((1, 1), dtype=complex)

    for site, A in enumerate(tensors):
        operator = operators.get(
            site, np.eye(A.shape[1], dtype=complex)
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


# Full-vector calculation, used only for comparison
def full_matrix_element(state, n_spins, operators=None):
    if operators is None:
        operators = {}

    operator = np.ones((1, 1), dtype=complex)

    for site in range(n_spins):
        local_operator = operators.get(
            site, np.eye(2, dtype=complex)
        )
        operator = np.kron(operator, local_operator)

    return np.vdot(state, operator @ state)



# State preparation
n_spins = 6
dimension = 2**n_spins

product = np.ones(dimension, dtype=complex) / np.sqrt(dimension)

ghz = np.zeros(dimension, dtype=complex)
ghz[0] = ghz[-1] = 1 / np.sqrt(2)

rng = np.random.default_rng(42)
random_state = (
    rng.normal(size=dimension)
    + 1j * rng.normal(size=dimension)
)
random_state /= np.linalg.norm(random_state)



# Local operators
X = np.array(
    [[0, 1],
     [1, 0]],
    dtype=complex,
)

Y = np.array(
    [[0, -1j],
     [1j, 0]],
    dtype=complex,
)

Z = np.array(
    [[1, 0],
     [0, -1]],
    dtype=complex,
)

i, j = 1, 4

observables = [
    (f"<X_{i}>", {i: X}),
    (f"<Y_{i}>", {i: Y}),
    (f"<Z_{i}>", {i: Z}),
    (f"<Z_{j}>", {j: Z}),
    (f"<Z_{i} Z_{j}>", {i: Z, j: Z}),
]



# Compare MPS contractions with full-vector calculations
for name, state in [
    ("Product", product),
    ("GHZ", ghz),
    ("Random", random_state),
]:
    tensors = state_to_mps(state, n_spins)

    mps_norm = mps_matrix_element(tensors)
    full_norm = np.vdot(state, state)

    print(f"\n{name}")
    print(f"  MPS norm squared:  {mps_norm.real:.12f}")
    print(f"  Full norm squared: {full_norm.real:.12f}")

    mps_values = {}
    full_values = {}

    for label, operators in observables:
        mps_value = (
            mps_matrix_element(tensors, operators) / mps_norm
        )
        full_value = (
            full_matrix_element(state, n_spins, operators)
            / full_norm
        )

        mps_values[label] = mps_value
        full_values[label] = full_value

        print(
            f"  {label:12s}"
            f" MPS: {mps_value.real: .8f}"
            f"  Full: {full_value.real: .8f}"
        )

    # Connected correlation:
    # <Z_i Z_j> - <Z_i><Z_j>
    mps_connected = (
        mps_values[f"<Z_{i} Z_{j}>"]
        - mps_values[f"<Z_{i}>"] * mps_values[f"<Z_{j}>"]
    )
    full_connected = (
        full_values[f"<Z_{i} Z_{j}>"]
        - full_values[f"<Z_{i}>"] * full_values[f"<Z_{j}>"]
    )

    print(
        f"  Connected ZZ:"
        f" MPS: {mps_connected.real: .8f}"
        f"  Full: {full_connected.real: .8f}"
    )
