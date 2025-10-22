import numpy as np

class MLP:
    def __init__(self, layer_sizes, learning_rate=0.01, momentum=0.9):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_layers = len(layer_sizes)

        self.weights = []
        self.biases = []
        self.weight_updates_prev = []
        self.loss_history = []

        for i in range(self.num_layers - 1):
            # Inicialização Xavier/Glorot melhorada
            scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            self.weight_updates_prev.append(np.zeros_like(w))

    # ===============================
    # Função de ativação sigmoide
    # ===============================
    @staticmethod
    def sigmoid(x):
        # Estabilidade numérica
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    # ===============================
    # Forward propagation
    # ===============================
    def forward(self, X):
        activations = [X]
        z_values = []

        for i in range(self.num_layers - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            z_values.append(z)
            activations.append(a)

        return activations, z_values

    # ===============================
    # Backpropagation CORRIGIDA
    # ===============================
    def backward(self, X, y, activations, z_values):
        deltas = [None] * (self.num_layers - 1)

        # Camada de saída - CORREÇÃO IMPORTANTE
        output = activations[-1]
        error = output - y  # MSE derivative: dL/dz = (y_hat - y) * sigmoid_derivative
        deltas[-1] = error * self.sigmoid_derivative(output)

        # Propagação do erro para trás - CORREÇÃO NO LOOP
        for l in range(self.num_layers - 3, -1, -1):  # Corrigido o range
            deltas[l] = np.dot(deltas[l+1], self.weights[l+1].T) * self.sigmoid_derivative(activations[l+1])

        # Atualização de pesos e biases
        for i in range(self.num_layers - 1):
            grad_w = np.dot(activations[i].T, deltas[i])
            grad_b = np.sum(deltas[i], axis=0, keepdims=True)

            # Momentum
            update = self.learning_rate * grad_w + self.momentum * self.weight_updates_prev[i]
            self.weights[i] -= update  # SUBTRAIR o gradiente
            self.biases[i] -= self.learning_rate * grad_b  # SUBTRAIR o gradiente

            self.weight_updates_prev[i] = update

    # ===============================
    # Treinamento
    # ===============================
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            activations, z_values = self.forward(X)
            self.backward(X, y.reshape(-1, 1), activations, z_values)

            loss = self.compute_loss(y.reshape(-1, 1), activations[-1])
            self.loss_history.append(loss)  # SALVAR HISTÓRICO

            if (epoch+1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    def train_mlp_with_split(self, X, y, test_size=0.2, epochs=500, random_seed=42):
        # Split treino/teste
        np.random.seed(random_seed)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        split_idx = int(X.shape[0] * (1 - test_size))

        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"Treino: X{X_train.shape}, y{y_train.shape}")
        print(f"Teste: X{X_test.shape}, y{y_test.shape}")

        # Treinamento
        self.train(X_train, y_train, epochs=epochs)

        return X_train, X_test, y_train, y_test

    # ===============================
    # Função de perda (Binary Cross-Entropy)
    # ===============================
    @staticmethod
    def compute_loss(y_true, y_pred):
        # Evitar log(0)
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # ===============================
    # Predição
    # ===============================
    def predict(self, X):
        activations, _ = self.forward(X)
        return (activations[-1] > 0.5).astype(int)

    # ===============================
    # Avaliação
    # ===============================
    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy