import numpy as np

up = np.array(
    [1, 0],
    dtype=complex
)
down = np.array(
    [0, 1],
    dtype=complex
)

up_up = np.kron(up, up)
up_down = np.kron(up, down)
down_up = np.kron(down, up)
down_down = np.kron(down, down)

print("Up-up:            ", up_up)
print("Up-down:          ", up_down)
print("Down-up:          ", down_up)
print("Down-down:        ", down_down)
print()

# First spin in superposition, second spin up
plus = (up + down) / np.sqrt(2)
product_state = np.kron(plus, up)

# Entangled state
bell_state = (up_up + down_down) / np.sqrt(2)

print("Product state:", product_state)
print("Bell state:   ", bell_state)
print()

print("Product norm squared:", np.vdot(product_state, product_state).real)
print("Bell norm squared:   ", np.vdot(bell_state, bell_state).real)
