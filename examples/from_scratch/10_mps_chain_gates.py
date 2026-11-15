import numpy as np



def mps_CNOT(tensors, CNOT, site_i, site_j, name):
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
        f"{name} shapes after CNOT ({site_i + 1}{site_j + 1}):",
        [A.shape for A in tensors]
    )


def truncate_bond(tensors, site, chi_max):
    theta = np.tensordot(tensors[site], tensors[site+1], axes=([-1], [0]))

    left_bond = theta.shape[0]
    right_bond = theta.shape[-1]
    matrix = theta.reshape(left_bond * 2, 2 * right_bond)

    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)

    bond = max(1, int(np.count_nonzero(s > 1e-12)))

    bond = min(bond, chi_max)   # <-- Truncation
    discarded_weight = (np.sum(s[bond:]**2) / np.sum(s**2))

    # Normalize the retained Schmidt coefficients.
    kept_s = s[:bond] / np.linalg.norm(s[:bond])

    tensors[site] = U[:, :bond].reshape(left_bond, 2, bond)
    tensors[site+1] = (np.diag(kept_s) @ Vh[:bond, :]).reshape(
        bond, 2, right_bond
    )

    return discarded_weight


def mps_to_state(tensors):
    state = tensors[0]
    for A in tensors[1:]:
        state = np.tensordot(state, A, axes=([-1], [0]))
    return state.reshape(-1)


def fidelity(tensors_a, tensors_b):
    a = mps_to_state(tensors_a)
    b = mps_to_state(tensors_b)

    return (
        abs(np.vdot(a, b))**2
        / (np.vdot(a, a).real * np.vdot(b, b).real)
    )


def expectation_Z(tensors, site):
    state = mps_to_state(tensors)
    amplitudes = state.reshape([2] * len(tensors))

    probabilities = np.abs(amplitudes)**2
    p0 = np.take(probabilities, 0, axis=site).sum()
    p1 = np.take(probabilities, 1, axis=site).sum()

    return (p0 - p1) / (p0 + p1)



# Setup
n_spins = 4

up = np.array(
    [1, 0],
    dtype=complex
)

tensors = [up.reshape(1, 2, 1).copy() for i in range(n_spins)]

print("============================================================================================================")
print("Initialization, rotation, and entanglement of sites 1 and 2")
print("============================================================================================================")
print("Initial tensor shapes:        ", [A.shape for A in tensors])



# Rotation acts on A1.
epsilon = 0.1

R = np.array(
    [[np.sqrt(1 - epsilon), -np.sqrt(epsilon)],
     [np.sqrt(epsilon),      np.sqrt(1 - epsilon)]],
    dtype=complex
)

tensors[0] = np.einsum("ij,ajb->aib", R, tensors[0])

print("Tensor shapes after rotating: ", [A.shape for A in tensors])



# CNOT entangles sites 1 and 2.
CNOT = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]],
     dtype=complex
)

mps_CNOT(tensors, CNOT, 0, 1, "Tensor")
print()



# Save three copies of the MPS for the three distinct treatments that will follow.
reference = [A.copy() for A in tensors]
truncate_once = [A.copy() for A in tensors]
truncate_repeatedly = [A.copy() for A in tensors]



# Truncate, rotate, and entangle.
for site in [0, 1, 2]:
    print("============================================================================================================")
    print(f"Truncation, rotation, and entanglement of sites {site} and {site+1}")
    print("============================================================================================================")

    if site == 0:
        w_once = truncate_bond(truncate_once, site=site, chi_max=1)
    truncation_site = max(0, site - 1)
    w_repeated = truncate_bond(truncate_repeatedly, site=truncation_site, chi_max=1)

    print("Fidelity, truncate once:",
        fidelity(reference, truncate_once))
    print("Fidelity, truncate repeatedly:",
        fidelity(reference, truncate_repeatedly))
    print("Weight discarded at this step:", w_repeated)
    print()

    for name, mps in [
        ("Reference", reference),
        ("Truncate once", truncate_once),
        ("Truncate repeatedly", truncate_repeatedly),
    ]:
        values = [expectation_Z(mps, site) for site in range(n_spins)]
        print(name, "<Zi>:", np.round(values, 6))
        print(name, "stored entries:", sum(A.size for A in mps))

        mps[site] = np.einsum("ij,ajb->aib", R, mps[site])
        mps_CNOT(mps, CNOT, site, site+1, name)
        print()



# Truncate "truncate_repeatedly".
print("============================================================================================================")
print("Final truncation")
print("============================================================================================================")
w_repeated = truncate_bond(truncate_repeatedly, site=2, chi_max=1)

print("Fidelity, truncate once:",
      fidelity(reference, truncate_once))
print("Fidelity, truncate repeatedly:",
      fidelity(reference, truncate_repeatedly))
print("Weight discarded at this step:", w_repeated)
print()

for name, mps in [
    ("Reference", reference),
    ("Truncate once", truncate_once),
    ("Truncate repeatedly", truncate_repeatedly),
]:
    values = [expectation_Z(mps, site) for site in range(n_spins)]
    print(name, "<Zi>:", np.round(values, 6))
    print(name, "stored entries:", sum(A.size for A in mps))
    print()
