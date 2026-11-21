from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensor_networks.mps import (
    state_to_mps,
    bond_dimensions,
    stored_entries
)


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

        bonds = bond_dimensions(tensors)
        mps_entries = stored_entries(tensors)

        print(f"\n  {name}")
        print("    Bond dimensions:", bonds)
        print("    Full-state entries:", state.size)
        print("    MPS entries:", mps_entries)
