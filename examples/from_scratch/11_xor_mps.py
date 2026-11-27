import numpy as np


def feature_map(x):
    angle = np.pi * x / 2

    return np.array([
        np.cos(angle),
        np.sin(angle)
    ])

def predict(x, A, B):
    phi_left = feature_map(x[0])
    phi_right = feature_map(x[1])

    left = phi_left @ A
    right = B @ phi_right

    return left @ right

def backprop(x, A, B):
    phi_left = feature_map(x[0])
    phi_right = feature_map(x[1])

    left = phi_left @ A
    right = B @ phi_right

    return np.outer(phi_left, right), np.outer(left, phi_right)

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

for bond in [1, 2]:
    A = rng.normal(scale=0.5, size=(2, bond))
    B = rng.normal(scale=0.5, size=(bond, 2))
    
    for epoch in range(epochs):
        loss = 0.0

        for n in range(N):
            x = X[n]

            # Forward pass
            f = predict(x, A, B)

            # Loss
            loss += mse_loss(y[n], f)

            # Backward pass
            dLdA = np.zeros_like(A)
            dLdB = np.zeros_like(B)

            dfdA, dfdB = backprop(x, A, B)

            delta = f - y[n]

            dLdA += delta * dfdA
            dLdB += delta * dfdB

            # Gradient descent update
            A -= lr * dLdA
            B -= lr * dLdB

        if epoch % 100 == 0:
            print(f"Bond: {bond} | Epoch: {epoch:5d} | Loss: {loss:.6f}")
    
    print()
