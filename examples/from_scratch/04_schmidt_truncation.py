import numpy as np
import matplotlib.pyplot as plt



# State preparation
up = np.array(
    [1, 0],
    dtype=complex
)
down = np.array(
    [0, 1],
    dtype=complex
)

epsilon = 0.05
state = np.sqrt(1 - epsilon) * np.kron(up, up) + np.sqrt(epsilon) * np.kron(down, down)

C = state.reshape(2, 2)
U, s, Vh = np.linalg.svd(C, full_matrices=False)
print("Schmidt coefficients:", s)



# Keep only the largest term
truncated_state = s[0] * np.kron(U[:, 0], Vh[0, :])

discarded_weight = np.sum(s[1:]**2)
truncation_error_squared = np.linalg.norm(state - truncated_state)**2

approx_state = truncated_state / np.linalg.norm(truncated_state)
fidelity = np.abs(np.vdot(state, approx_state))**2

print("Discarded weight:", discarded_weight)
print("Squared error before normalization:", truncation_error_squared)
print("Fidelity after normalization:", fidelity)
print()



# Compare an expectation value
sigma_z = np.array(
    [[1, 0],
    [0, -1]],
    dtype=complex
)
identity = np.eye(2, dtype=complex)

O = np.kron(sigma_z, identity)

exact_expectation = np.vdot(state, O @ state).real
approx_expectation = np.vdot(approx_state, O @ approx_state).real

observable_error = abs(exact_expectation - approx_expectation)

print("Exact expectation:", exact_expectation)
print("Approximate expectation:", approx_expectation)
print("Observable error:", observable_error)



# Repeat for different values of epsilon
epsilons = np.linspace(0, 0.5, 100)

sigma_x = np.array(
    [[0, 1],
    [1, 0]],
    dtype=complex
)
observables = [
    np.kron(sigma_z, identity),
    np.kron(sigma_z, sigma_z),
    np.kron(sigma_x, sigma_x)
]

colors = ['black', 'red', 'blue']

labels = [r'$\sigma_z\otimes I$', r'$\sigma_z\otimes\sigma_z$', r'$\sigma_x\otimes\sigma_x$']

for label, color, observable in zip(labels, colors, observables):
    for eps in epsilons:
        state = np.sqrt(1 - eps) * np.kron(up, up) + np.sqrt(eps) * np.kron(down, down)
        C = state.reshape(2, 2)
        U, s, Vh = np.linalg.svd(C, full_matrices=False)

        truncated_state = s[0] * np.kron(U[:, 0], Vh[0, :])
        approx_state = truncated_state / np.linalg.norm(truncated_state)

        exact_expectation = np.vdot(state, observable @ state).real
        approx_expectation = np.vdot(approx_state, observable @ approx_state).real
        observable_error = abs(exact_expectation - approx_expectation)

        plt.scatter(eps, observable_error, color=color, s=1, label=r'$\mathcal{O}=$' + label if eps == epsilons[0] else '')

plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$|\delta\langle\mathcal{O}\rangle|$')
plt.legend(frameon=False)
plt.show()
