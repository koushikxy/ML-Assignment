import numpy as np
import matplotlib.pyplot as plt

class SingleLayerPerceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            for i in range(n_samples):
                # Compute the predicted value
                predicted = self.predict(X[i])

                # Update weights and bias
                self.weights += self.learning_rate * (y[i] - predicted) * X[i]
                self.bias += self.learning_rate * (y[i] - predicted)

    def predict(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return 1 if linear_output >= 0 else 0

# NAND dataset
X_nand = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_nand = np.array([1, 1, 1, 0])

# NOR dataset
X_nor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_nor = np.array([1, 0, 0, 0])

# AND dataset
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# OR dataset
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

# Function to plot separation line
def plot_separation_line(X, y, perceptron, title):
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='1')
    plt.title(title)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    
    # Plot separation line
    x_vals = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
    y_vals = -(perceptron.weights[0] * x_vals + perceptron.bias) / perceptron.weights[1]
    plt.plot(x_vals, y_vals, label='Separation line')
    
    plt.legend()
    plt.show()

# Train and plot for NAND dataset
perceptron_nand = SingleLayerPerceptron()
perceptron_nand.fit(X_nand, y_nand)
plot_separation_line(X_nand, y_nand, perceptron_nand, 'NAND Dataset')

# Train and plot for NOR dataset
perceptron_nor = SingleLayerPerceptron()
perceptron_nor.fit(X_nor, y_nor)
plot_separation_line(X_nor, y_nor, perceptron_nor, 'NOR Dataset')

# Train and plot for AND dataset
perceptron_and = SingleLayerPerceptron()
perceptron_and.fit(X_and, y_and)
plot_separation_line(X_and, y_and, perceptron_and, 'AND Dataset')

# Train and plot for OR dataset
perceptron_or = SingleLayerPerceptron()
perceptron_or.fit(X_or, y_or)
plot_separation_line(X_or, y_or, perceptron_or, 'OR Dataset')
