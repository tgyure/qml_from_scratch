import numpy as np


def RY(theta):
    return np.array([
        [np.cos(theta/2), np.sin(-theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ])

def CNOT(state):
    gate = np.array(
        [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]],
        dtype=complex
    )

    return gate @ state

def circuit(x, theta1, theta2, theta3, use_cnot):
    up = np.array([1, 0], dtype=complex)

    Z = np.array([
        [1, 0],
        [0, -1]
    ])

    Z2 = np.kron(np.eye(2), Z)

    # Encoding layer
    state = np.kron(
        RY(np.pi * x[0]) @ up,
        RY(np.pi * x[1]) @ up
    )

    # Layer 1
    state = np.kron(RY(theta1), RY(theta2)) @ state

    # Layer 2
    if use_cnot:
        state = CNOT(state)

    # Layer 3
    state = np.kron(np.eye(2), RY(theta3)) @ state

    return np.vdot(state, Z2 @ state).real

def mse_loss(y, y_hat):
    return 0.5 * np.mean((y_hat - y) ** 2)


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([0, 1, 1, 0], dtype=float)

N = X.shape[0]


rng = np.random.default_rng(42)
epochs = 1000
lr = 0.5


for use_cnot in [False, True]:
    theta1, theta2, theta3 = rng.uniform(-2.5, 2.5, 3)

    for epoch in range(epochs):
        loss = 0.0

        for n in range(N):
            x = X[n]

            # Forward pass
            z = circuit(x, theta1, theta2, theta3, use_cnot)
            y_hat = (1 - z) / 2

            # Loss
            loss += mse_loss(y[n], y_hat)

            # Backward pass
            delta = (y_hat - y[n]) * (-0.5)

            s = np.pi / 2
            dz1 = (
                circuit(x, theta1 + s, theta2, theta3, use_cnot)
                - circuit(x, theta1 - s, theta2, theta3, use_cnot)
            ) / 2
            dz2 = (
                circuit(x, theta1, theta2 + s, theta3, use_cnot)
                - circuit(x, theta1, theta2 - s, theta3, use_cnot)
            ) / 2
            dz3 = (
                circuit(x, theta1, theta2, theta3 + s, use_cnot)
                - circuit(x, theta1, theta2, theta3 - s, use_cnot)
            ) / 2

            dtheta1 = delta * dz1
            dtheta2 = delta * dz2
            dtheta3 = delta * dz3

            # Gradient descent update
            theta1 -= lr * dtheta1
            theta2 -= lr * dtheta2
            theta3 -= lr * dtheta3

        if epoch % 100 == 0:
            print(f"CNOT: {use_cnot} | Epoch: {epoch:5d} | Loss: {loss:.6f}")

    print()
