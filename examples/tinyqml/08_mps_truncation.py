from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensor_networks.mps import state_to_mps, stored_entries
from tensor_networks.contractions import mps_expectation, mps_fidelity
from tensor_networks.canonical import truncate_bond



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

for (name, state), color, marker, marker_size in zip(states, colors, markers, marker_sizes):
    print(f"\n{name}")

    exact_values = [
        full_matrix_element(state, n_spins, operators).real
        for label, operators in observables
    ]

    storage_ratios = []
    fidelities = []
    observable_errors = [[] for _ in observables]

    reference = state_to_mps(state, n_spins)

    for max_bond in bond_caps:
        tensors = [A.copy() for A in reference]

        # Truncate each internal bond.
        for site in range(n_spins - 1):
            truncate_bond(tensors, site=site, max_bond=max_bond)

        # State fidelity and storage.
        fidelity = mps_fidelity(reference, tensors)
        storage_ratio = stored_entries(tensors) / state.size

        fidelities.append(fidelity)
        storage_ratios.append(storage_ratio)

        print(f"\n  Maximum bond dimension: {max_bond}")
        print(f"  MPS/full entries: {storage_ratio:.4f}")
        print(f"  Fidelity: {fidelity:.8f}")

        # Observable errors.
        for index, ((label, operators), exact) in enumerate(
            zip(observables, exact_values)
        ):
            approximate = mps_expectation(tensors, operators).real

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
