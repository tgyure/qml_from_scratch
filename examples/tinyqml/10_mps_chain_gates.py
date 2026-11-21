from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensor_networks.mps import stored_entries
from tensor_networks.canonical import truncate_bond
from tensor_networks.contractions import mps_expectation, mps_fidelity
from tensor_networks.gates import one_site_gate, two_site_gate



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

one_site_gate(tensors, R, 0)

print("Tensor shapes after rotating: ", [A.shape for A in tensors])



# CNOT entangles sites 1 and 2.
CNOT = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]],
     dtype=complex
)

two_site_gate(tensors, CNOT, 0)
print()



# Operators for expectation values
Z = np.array(
    [[1, 0],
     [0, -1]],
    dtype=complex,
)



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
        w_once = truncate_bond(truncate_once, site=site, max_bond=1)
    truncation_site = max(0, site - 1)
    w_repeated = truncate_bond(truncate_repeatedly, site=truncation_site, max_bond=1)

    print("Fidelity, truncate once:",
        mps_fidelity(reference, truncate_once))
    print("Fidelity, truncate repeatedly:",
        mps_fidelity(reference, truncate_repeatedly))
    print("Weight discarded at this step:", w_repeated)
    print()

    for name, mps in [
        ("Reference", reference),
        ("Truncate once", truncate_once),
        ("Truncate repeatedly", truncate_repeatedly),
    ]:
        values = [mps_expectation(mps, {site: Z}).real  for site in range(n_spins)]
        print(name, "<Zi>:", np.round(values, 6))
        print(name, "stored entries:", stored_entries(mps))

        one_site_gate(mps, R, site)
        two_site_gate(mps, CNOT, site)
        print()



# Truncate "truncate_repeatedly".
print("============================================================================================================")
print("Final truncation")
print("============================================================================================================")
w_repeated = truncate_bond(truncate_repeatedly, site=2, max_bond=1)

print("Fidelity, truncate once:",
      mps_fidelity(reference, truncate_once))
print("Fidelity, truncate repeatedly:",
      mps_fidelity(reference, truncate_repeatedly))
print("Weight discarded at this step:", w_repeated)
print()

for name, mps in [
    ("Reference", reference),
    ("Truncate once", truncate_once),
    ("Truncate repeatedly", truncate_repeatedly),
]:
    values = [mps_expectation(mps, {site: Z}).real  for site in range(n_spins)]
    print(name, "<Zi>:", np.round(values, 6))
    print(name, "stored entries:", stored_entries(mps))
    print()
