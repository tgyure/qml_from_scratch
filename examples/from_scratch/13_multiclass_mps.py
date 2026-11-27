import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


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
    right = np.einsum("bim,i->bm", B, phi_right)

    return left @ right

def backprop(x, A, B, d, delta):
    phi_left = feature_map(x[0], d)
    phi_right = feature_map(x[1], d)

    left = phi_left @ A
    right = np.einsum("bim,i->bm", B, phi_right)


    dfdA = [np.outer(phi_left, right[:, i]) for i in range(B.shape[2])]
    dfdB = np.outer(left, phi_right)

    return dfdA, dfdB

def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def mse_loss(y, y_hat):
    return 0.5 * np.mean((y_hat - y) ** 2)


num_clusters = 3

X, y = make_blobs(
    n_samples=600,
    centers=num_clusters,
    cluster_std=0.5,
    random_state=42
)
X = (X - X.min(0)) / (X.max(0) - X.min(0))

N = X.shape[0]


rng = np.random.default_rng(42)
epochs = 100
lr = 0.05

feature_size = 5
bond = 2

A = rng.normal(scale=0.5, size=(feature_size, bond))
B = rng.normal(scale=0.5, size=(bond, feature_size, num_clusters))

for epoch in range(epochs):
    loss = 0.0

    for n in range(N):
        x = X[n]

        # Forward pass
        f = predict(x, A, B, feature_size)
        p = softmax(f)

        # Loss
        loss += -np.log(p[y[n]])

        # Backward pass
        dLdA = np.zeros_like(A)
        dLdB = np.zeros_like(B)

        delta = p - np.eye(num_clusters)[y[n]]
        dfdA, dfdB = backprop(x, A, B, feature_size, delta)

        for i in range(num_clusters):
            dLdA += delta[i] * dfdA[i]
            dLdB[:, :, i] += delta[i] * dfdB

        # Gradient descent update
        A -= lr * dLdA
        B -= lr * dLdB

    if epoch % 10 == 0:
        print(f"Epoch: {epoch:5d} | Loss: {loss:.6f}")


xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

scores = np.array([predict(p, A, B, feature_size) for p in grid])
Z = np.argmax(scores, axis=1).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(num_clusters + 1) - 0.5)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
plt.title("Multiclass MPS Classifier")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
