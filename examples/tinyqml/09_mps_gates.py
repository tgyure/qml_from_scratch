from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensor_networks.gates import one_site_gate, two_site_gate



# Setup
n_spins = 4

up = np.array(
    [1, 0],
    dtype=complex
)

tensors = [up.reshape(1, 2, 1).copy() for i in range(n_spins)]

print("Initial tensor shapes:        ", [A.shape for A in tensors])



# Hadamard acts on A1.
H = np.array(
    [[1, 1],
    [1, -1]]
) / np.sqrt(2)

one_site_gate(tensors, H, 0)

print("Tensor shapes after Hadamard: ", [A.shape for A in tensors])
print()



# CNOT entangles sites 1 and 2, then 2 and 3, and finally 3 and 4.
CNOT = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]],
     dtype=complex
)

two_site_gate(tensors, CNOT, 0)
print("Tensor shapes after CNOT (01):", [A.shape for A in tensors])

two_site_gate(tensors, CNOT, 1)
print("Tensor shapes after CNOT (12):", [A.shape for A in tensors])

two_site_gate(tensors, CNOT, 2)
print("Tensor shapes after CNOT (23):", [A.shape for A in tensors])
