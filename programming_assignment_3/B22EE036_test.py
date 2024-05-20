# test.py
import numpy as np
import argparse

def load_weights(filepath='weights.txt'):
    """Load weights from a file."""
    with open(filepath, 'r') as f:
        weights = np.array([float(weight) for weight in f.readlines()])
    return weights

def load_data(filepath):
    """Load test data."""
    return np.loadtxt(filepath, delimiter=',')

def predict(X, weights):
    """Predict labels for the test data."""
    """Predict labels for the test data."""
    bias = weights[0]    
    weights_without_bias = weights[1:]    

    if X.ndim == 1:
        X = X.reshape(1, -1)  # Workaround for single sample prediction, else there's a dimension error.

    dot_product = np.dot(X, weights_without_bias) + bias    
    return np.where(dot_product >= 0, 1, 0)                 #conditional prediction

def main(test_file):
    weights = load_weights()
    X_test = load_data(test_file)
    predictions = predict(X_test, weights)
    print(','.join(map(str, predictions)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a perceptron model.")
    parser.add_argument('test_file', help='File path to the test dataset.', type=str)
    args = parser.parse_args()

    main(args.test_file)
