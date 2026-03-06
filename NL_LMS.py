import numpy as np

class LMS:
    def __init__(self, n_neurons, input_size, learning_rate=0.01):
        self.n_neurons = n_neurons
        self.input_size = input_size
        self.mu = learning_rate

        self.w_ij = np.random.normal(0, .01, (n_neurons, input_size))

    def s(self, x):
        x = np.array(x, dtype=float)
        return np.dot(self.w_ij, x)
    
    def learn(self, x, target):
        x = np.array(x, dtype=float)
        target = np.array(target, dtype=float)

        pred = self.s(x)

        error = target - pred

        norm_x = np.dot(x, x)
        epsilon = 1e-6
        normalization = 1.0 / (norm_x + epsilon)
        delta_w = self.mu * normalization * np.outer(error, x)

        self.w_ij += delta_w
        return np.mean(np.abs(error))