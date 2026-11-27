import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


def feature_map(x, d):
    features = [np.ones_like(x)]
    k = 1

    while len(features) < d:
        features.append(np.cos(k * np.pi * x))
        if len(features) < d:
            features.append(np.sin(k * np.pi * x))
        k += 1
    
    return np.array(features)

def predict(x, A, B, d):
    phi_left = feature_map(x[0], d)
    phi_right = feature_map(x[1], d)

    left = phi_left @ A
    right = B @ phi_right

    return left @ right

def backprop(x, A, B, d):
    phi_left = feature_map(x[0], d)
    phi_right = feature_map(x[1], d)

    left = phi_left @ A
    right = B @ phi_right

    return np.outer(phi_left, right), np.outer(left, phi_right)

def mse_loss(y, y_hat):
    return 0.5 * np.mean((y_hat - y) ** 2)


X, y = make_moons(
    n_samples=500,
    noise=0.2,
    random_state=0
)
X = (X - X.min(0)) / (X.max(0) - X.min(0))

N = X.shape[0]


fig, axes = plt.subplots(3, 3, figsize=(12, 12))
xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
grid = np.c_[xx.ravel(), yy.ravel()]


rng = np.random.default_rng(42)
epochs = 100
lr = 0.2

for row, feature_size in enumerate([2, 5, 9]):

    for bond in [1, 2, 3]:
        if bond > feature_size:
            continue

        A = rng.normal(scale=0.5, size=(feature_size, bond))
        B = rng.normal(scale=0.5, size=(bond, feature_size))
        
        for epoch in range(epochs):
            loss = 0.0

            for n in range(N):
                x = X[n]

                # Forward pass
                f = predict(x, A, B, feature_size)

                # Loss
                loss += mse_loss(y[n], f)

                # Backward pass
                dLdA = np.zeros_like(A)
                dLdB = np.zeros_like(B)

                dfdA, dfdB = backprop(x, A, B, feature_size)

                delta = f - y[n]

                dLdA += delta * dfdA
                dLdB += delta * dfdB

                # Gradient descent update
                A -= (lr / feature_size) * dLdA
                B -= (lr / feature_size) * dLdB

            if epoch % 10 == 0:
                print(f"Feature size: {feature_size} | Bond: {bond} | Epoch: {epoch:5d} | Loss: {loss:.6f}")
        
        ax = axes[row, bond - 1]
        f_grid = np.array([predict(p, A, B, feature_size) for p in grid]).reshape(xx.shape)
        ax.contourf(xx, yy, f_grid, levels=50)
        ax.contour(xx, yy, f_grid, levels=[0.5], colors="k")
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=10)
        ax.set_title(f"feature size {feature_size}, bond {bond}")
        print()

axes[0, 2].axis("off")
plt.tight_layout()
plt.show()
