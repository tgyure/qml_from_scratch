import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def feature_map(x):
    angle = np.pi * x / 2

    return np.stack([
        np.cos(angle),
        np.sin(angle)
    ], axis=-1)

def predict(x, tensors):
    phi = feature_map(x)

    left = phi[0] @ tensors[0]
    right = np.einsum("bim,i->bm", tensors[-1], phi[-1])
    
    for i in range(1, len(x) - 1):
        middle = phi[i] @ tensors[i]
        left = left @ middle

    return left @ right

def backprop(x, tensors):
    phi = feature_map(x)
    n = len(tensors)
    M = [phi[i] @ tensors[i] for i in range(n - 1)] + [np.einsum("bim,i->bm", tensors[-1], phi[-1])]

    lefts = [None] * n
    L = M[0]
    for i in range(1, n):
        lefts[i] = L
        if i < n - 1:
            L = L @ M[i]
    
    rights = [None] * n
    R = M[-1]
    for i in range(n - 2, -1, -1):
        rights[i] = R
        if i > 0:
            R = M[i] @ R
    
    dA = [None] * n
    dA[0] = np.einsum("i,bm->ibm", phi[0], rights[0])
    for i in range(1, n - 1):
        dA[i] = np.einsum("a,i,bm->aibm", lefts[i], phi[i], rights[i])
    dA[-1] = np.einsum("a,i->ai", lefts[-1], phi[-1])

    return dA

def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


digits = load_digits()
X = digits.images.reshape(len(digits.images), -1) / 16.0
y = digits.target

rng = np.random.default_rng(0)

indices = np.concatenate([
    rng.choice(np.where(y == digit)[0], size=50, replace=False)
    for digit in range(10)
])

rng.shuffle(indices)
X = X[indices]
y = y[indices]

N = len(X)


rng = np.random.default_rng(42)
epochs = 100
lr = 1e-4

bond = 6
I = np.eye(bond)
tensors = (
    [np.eye(2, bond) + 0.01 * rng.normal(size=(2, bond))]
    + [np.stack([I, I], axis=1) + 0.01 * rng.normal(size=(bond, 2, bond)) for _ in range(62)]
    + [0.01 * rng.normal(size=(bond, 2, 10))]
)

for epoch in range(epochs):
    loss = 0.0

    for n in range(N):
        x = X[n]

        # Forward pass
        f = predict(x, tensors)
        p = softmax(f)

        # Loss
        loss += -np.log(p[y[n]])

        # Backward pass
        dA = backprop(x, tensors)

        delta = p - np.eye(10)[y[n]]

        # Gradient descent update
        for i in range(len(tensors) - 1):
            tensors[i] -= lr * dA[i] @ delta
        tensors[-1] -= lr * np.einsum("ai,m->aim", dA[-1], delta)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch:5d} | Loss: {loss:.6f}")


test_indices = np.setdiff1d(np.arange(len(digits.target)), indices)
sample_indices = rng.choice(test_indices, size=20, replace=False)

X_plot = digits.images[sample_indices] / 16.0
y_plot = digits.target[sample_indices]

predictions = [np.argmax(predict(image.reshape(-1), tensors)) for image in X_plot]

fig, axes = plt.subplots(4, 5, figsize=(10, 8))

for ax, image, truth, prediction in zip(axes.ravel(), X_plot, y_plot, predictions):
    ax.imshow(image, cmap="gray")
    ax.set_title(
        f"True: {truth}, Pred: {prediction}",
        color="green" if truth == prediction else "red"
    )
    ax.axis("off")

plt.tight_layout()
plt.show()
