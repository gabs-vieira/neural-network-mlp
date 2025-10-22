import numpy as np

class MLP:
    def __init__(self, layer_sizes, learning_rate=0.01, momentum=0.9):
        """
        Args
            :layer_sizes: lista com o número de neurônios por camada, incluindo input e output. Ex: [n_features, 10, 5, 1] -> 2 camadas ocultas + saída
            :learning_rate: taxa de aprendizado (alpha)
            :momentum: fator de momentum
        """

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_layers = len(layer_sizes)


        # Inicializar pesos e biases (pequenos valores aleatórios)
        self.weights = []
        self.biases = []
        self.weight_updates_prev = []  # Para momentum
        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            self.weight_updates_prev.append(np.zeros_like(w))


    # ===============================
    # Função de ativação sigmoide
    # ===============================
    @staticmethod
    def sigmoid(x):
        return 1 / (1+np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)


    # ===============================
    # Forward propagation
    # ===============================
    def forward(self, X):
        activations = [X] #lista com ativacao de cada camada
        z_values = [] #valores antes da ativacao

        for i in range(self.num_layers - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            z_values.append(z)
            activations.append(a)

        return activations, z_values


    # ===============================
    # Backpropagation
    # ===============================
    def backward(self, X, y, activations, z_values):
        deltas = [None] * (self.num_layers - 1)  # delta por camada

        # Camada de saída (Binary Cross-Entropy derivada + sigmoide)
        output = activations[-1]
        # FatorErro = y - y_hat
        factor_error = y - output
        deltas[-1] = factor_error * self.sigmoid_derivative(output)

        # Camadas ocultas
        for l in reversed(range(self.num_layers - 2)):
            deltas[l] = self.sigmoid_derivative(activations[l+1]) * np.dot(deltas[l+1], self.weights[l+1].T)

        # Atualização de pesos e biases
        for i in range(self.num_layers - 1):
            grad_w = np.dot(activations[i].T, deltas[i])
            grad_b = np.sum(deltas[i], axis=0, keepdims=True)

            # Momentum
            update = self.learning_rate * grad_w + self.momentum * self.weight_updates_prev[i]
            self.weights[i] += update
            self.biases[i] += self.learning_rate * grad_b

            self.weight_updates_prev[i] = update


    # ===============================
    # Treinamento
    # ===============================
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            activations, z_values = self.forward(X)
            self.backward(X, y, activations, z_values)

            if (epoch+1) % 100 == 0 or epoch == 0:
                loss = self.compute_loss(y, activations[-1])
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")


    def train_mlp_with_split(self, X, y, test_size=0.2, epochs=500, random_seed=42):
        # ===============================
        # 1. Split treino/teste
        # ===============================
        np.random.seed(random_seed)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        split_idx = int(X.shape[0] * (1 - test_size))

        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ===============================
        # 2. Treinamento
        # ===============================
        self.train(X_train, y_train, epochs=epochs)

        # ===============================
        # 3. Retornar dados de teste para avaliação
        # ===============================
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
    # Avaliacao
    # ===============================

