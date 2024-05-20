import argparse
import numpy as np

class Perceptron:
    def __init__(self, X_train, y, weights=4, learning_rate=1, iterations=1000):
        self.X_train = self.normalize(X_train)       # normalizing step.
        self.y = y
        self.lr = learning_rate
        self.iter = iterations
        self.weights = np.random.randn(weights + 1)  # Including the bias term w0
        self.X_train_with_intercept = np.hstack((np.ones((self.X_train.shape[0], 1)), self.X_train))

    def train(self):
        print("Weights : ", self.weights)
        print("Train :", self.X_train_with_intercept)
        iters = 0
        while not self.convergence() and iters < self.iter:
            print(self.convergence())
            iters += 1
            for i, (x, label) in enumerate(zip(self.X_train_with_intercept, self.y)):
                y_pred = self.predict(self.weights, x)
                if label == 0 and y_pred != 0:
                    self.weights = self.weights - self.lr * x
                elif label == 1 and y_pred != 1:
                    self.weights = self.weights + self.lr * x
        print(f"Iters: {iters}, Accuracy: {self.accuracy()}")

    def convergence(self):
        if self.accuracy() == 1.0:
          return True
        return False

    def accuracy(self):
        correct_predictions = 0
        for x, label in zip(self.X_train_with_intercept, self.y):
            y_pred = self.predict(self.weights, x)
            if y_pred == label:
                correct_predictions += 1
        return correct_predictions / len(self.y)

    def predict(self, w, feature):
        # print("w", w)
        # print("feeeee", feature)
        # print(np.dot(w, feature))
        t = np.dot(w, feature)
        return 1 if t >= 0 else 0

    def normalize(self, X):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        return (X - mean) / std

    def save_weights(self):
        weights_str = '\n'.join(map(str, self.weights))
        with open('weights.txt', 'w') as file:
            file.write(weights_str)

def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    return X, y

def main():
    parser = argparse.ArgumentParser(description='Train a perceptron model.')
    parser.add_argument('train_file', type=str, help='Path to the training data file.')
    args = parser.parse_args()

    X_train, y_train = load_data(args.train_file)
    perceptron = Perceptron(X_train, y_train, weights=X_train.shape[1])
    perceptron.train()
    perceptron.save_weights()
    print("Training Over and Weights are saved")

if __name__ == "__main__":
    main()