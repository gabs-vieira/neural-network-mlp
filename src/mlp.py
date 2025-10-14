import numpy as np

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def sigmoid_deriv(a): return a * (1 - a)

class SimpleMLP:
    def __init__(self, layer_sizes, lr=0.05, momentum=0.9):
        self.sizes = layer_sizes
        self.lr, self.momentum = lr, momentum
        self.weights, self.biases, self.vel_w, self.vel_b = [], [], [], []
        for i in range(len(layer_sizes) - 1):
            in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (in_size + out_size))
            W = np.random.uniform(-limit, limit, (out_size, in_size))
            b = np.zeros((out_size, 1))
            self.weights.append(W); self.biases.append(b)
            self.vel_w.append(np.zeros_like(W)); self.vel_b.append(np.zeros_like(b))

    def forward(self, X):
        a = X.T; activations = [a]
        for W, b in zip(self.weights, self.biases):
            z = W.dot(a) + b; a = sigmoid(z); activations.append(a)
        return activations

    def predict_proba(self, X): return self.forward(X)[-1].T[:, 0]
    def predict(self, X, th=0.5): return (self.predict_proba(X) >= th).astype(int)

    def fit(self, X, y, epochs=100, batch_size=32, verbose=True):
        n = X.shape[0]; history = {'loss': []}
        for ep in range(epochs):
            perm = np.random.permutation(n); Xs, ys = X[perm], y[perm]
            for i in range(0, n, batch_size):
                xb, yb = Xs[i:i + batch_size], ys[i:i + batch_size]
                self._update_batch(xb, yb)
            yhat = self.predict_proba(X); loss = np.mean((y - yhat) ** 2)
            history['loss'].append(loss)
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"época {ep + 1}/{epochs} - loss: {loss:.6f}")
        return history

    def _update_batch(self, Xb, yb):
        Xb, yb = Xb.T, yb.reshape(1, -1)
        acts = [Xb]
        for W, b in zip(self.weights, self.biases):
            z = W.dot(acts[-1]) + b; a = sigmoid(z); acts.append(a)
        deltas = [None] * len(self.weights)
        a_out = acts[-1]
        deltas[-1] = sigmoid_deriv(a_out) * (yb - a_out)
        for l in range(len(self.weights) - 2, -1, -1):
            W_next, delta_next = self.weights[l + 1], deltas[l + 1]
            a_l = acts[l + 1]
            deltas[l] = sigmoid_deriv(a_l) * (W_next.T.dot(delta_next))
        m = Xb.shape[1]
        for l in range(len(self.weights)):
            a_prev = acts[l]
            grad_W = deltas[l].dot(a_prev.T) / m
            grad_b = np.mean(deltas[l], axis=1, keepdims=True)
            self.vel_w[l] = self.momentum * self.vel_w[l] + self.lr * grad_W
            self.vel_b[l] = self.momentum * self.vel_b[l] + self.lr * grad_b
            self.weights[l] += self.vel_w[l]
            self.biases[l] += self.vel_b[l]
