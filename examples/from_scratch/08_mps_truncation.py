import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



def state_to_mps(state, n_spins, tol=1e-12, max_bond=None):
    tensors = []
    remainder = state.reshape(1, -1)
    left_bond = 1

    for site in range(n_spins - 1):
        matrix = remainder.reshape(left_bond * 2, -1)
        U, s, Vh = np.linalg.svd(matrix, full_matrices=False)

        right_bond = max(1, int(np.count_nonzero(s > tol)))
        if max_bond is not None:
            right_bond = min(right_bond, max_bond)

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



n_spins = 8
dimension = 2**n_spins

rng = np.random.default_rng(42)

# Random state
random_state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
random_state /= np.linalg.norm(random_state)

# Product state: |+> tensor ... tensor |+>
product = np.ones(dimension, dtype=complex) / np.sqrt(dimension)

# GHZ state
ghz = np.zeros(dimension, dtype=complex)
ghz[0] = ghz[-1] = 1 / np.sqrt(2)

# Compressible state:
# |po> + 1/2 |p1> + 1/4 |p2> + ...,
# where |pk> is a normalized product state.
compressible_state = np.zeros(dimension, dtype=complex)

for term in range(8):
    product_state = np.ones(1, dtype=complex)

    for site in range(n_spins):
        spin = rng.normal(size=2) + 1j * rng.normal(size=2)
        spin /= np.linalg.norm(spin)

        product_state = np.kron(product_state, spin)

    compressible_state += 0.5**term * product_state

compressible_state /= np.linalg.norm(compressible_state)



states = [
    ("Product", product),
    ("GHZ", ghz),
    ("Compressible", compressible_state),
    ("Random", random_state),
]

bond_caps = [1, 2, 4, 8, 16]

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

observables = [
    (r"$X_0$", {0: X}),
    (r"$Z_0$", {0: Z}),
    (r"$Z_0 Z_1$", {0: Z, 1: Z}),
]



colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
markers = ["o", "s", "^", "D"]
marker_sizes = [8, 7, 6, 5]
linestyles = ["-", "--", ":"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Only for displaying zero or roundoff-scale errors on a log axis.
error_floor = 1e-15

for (name, state), color, marker, marker_size in zip(
    states, colors, markers, marker_sizes
):
    print(f"\n{name}")

    exact_values = [
        full_matrix_element(state, n_spins, operators).real
        for label, operators in observables
    ]

    storage_ratios = []
    fidelities = []
    observable_errors = [[] for _ in observables]

    for max_bond in bond_caps:
        tensors = state_to_mps(
            state, n_spins, max_bond=max_bond
        )

        # Normalize after truncation.
        norm_squared = mps_matrix_element(tensors).real
        tensors[-1] /= np.sqrt(norm_squared)

        # State fidelity and storage.
        approx_state = mps_to_state(tensors)
        fidelity = np.abs(np.vdot(state, approx_state))**2

        mps_entries = sum(A.size for A in tensors)
        storage_ratio = mps_entries / state.size

        storage_ratios.append(storage_ratio)
        fidelities.append(fidelity)

        print(f"\n  Maximum bond dimension: {max_bond}")
        print(f"  MPS/full entries: {storage_ratio:.4f}")
        print(f"  Fidelity: {fidelity:.8f}")

        # Observable errors.
        for index, ((label, operators), exact) in enumerate(
            zip(observables, exact_values)
        ):
            approximate = mps_matrix_element(
                tensors, operators
            ).real

            error = abs(approximate - exact)
            observable_errors[index].append(error)

            print(
                f"  {label}:"
                f" exact = {exact:.6f},"
                f" approximate = {approximate:.6f},"
                f" error = {error:.3e}"
            )

    plot_style = dict(
        color=color,
        marker=marker,
        markersize=marker_size,
        markerfacecolor="none",
        markeredgewidth=1,
        linewidth=1,
    )

    axes[0].plot(
        bond_caps, storage_ratios,
        linestyle="-", label=name, **plot_style,
    )
    axes[1].plot(
        bond_caps, fidelities,
        linestyle="-", **plot_style,
    )

    for errors, linestyle in zip(observable_errors, linestyles):
        axes[2].plot(
            bond_caps,
            np.maximum(errors, error_floor),
            linestyle=linestyle,
            **plot_style,
        )


axes[0].set_ylabel("MPS entries / full-state entries")
axes[0].axhline(1, color="gray", linestyle="--", linewidth=1)
axes[0].legend(title="State", frameon=False)

axes[1].set_ylabel("Fidelity")
axes[1].set_ylim(-0.03, 1.03)

axes[2].set_ylabel("Absolute observable error")
axes[2].set_yscale("log")
axes[2].set_title(r"Errors below $10^{-15}$ shown at $10^{-15}$")

observable_handles = [
    Line2D(
        [0], [0],
        color="black",
        linestyle=linestyle,
        linewidth=1,
        label=label,
    )
    for (label, _), linestyle in zip(observables, linestyles)
]
axes[2].legend(
    handles=observable_handles, title="Observable", frameon=False
)

for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks(bond_caps)
    ax.set_xticklabels([str(cap) for cap in bond_caps])
    ax.set_xlabel("Maximum bond dimension")
    ax.grid(False, which="both", axis="both")
    ax.tick_params(
        axis="both", which="both", direction="in", top=True, right=True,
        bottom=True, left=True,
    )

fig.tight_layout()
plt.show()
