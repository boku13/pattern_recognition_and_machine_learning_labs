import csv
import numpy as np


def ComputeMeanDiff(X):
    # Separate the samples by class
    class_zero = X[X[:, 2] == 0][:, :2]
    class_one = X[X[:, 2] == 1][:, :2]
    # means of each class
    mean_zero = np.mean(class_zero, axis=0)
    mean_one = np.mean(class_one, axis=0)
    # difference of class wise means
    mean_diff = mean_zero - mean_one
    return mean_diff


def ComputeSW(X):
    # Separate the samples by class
    class_zero = X[X[:, 2] == 0][:, :2]
    class_one = X[X[:, 2] == 1][:, :2]
    # scatter matrices for each class
    S_zero = np.dot((class_zero - np.mean(class_zero, axis=0)).T,
                    (class_zero - np.mean(class_zero, axis=0)))
    S_one = np.dot((class_one - np.mean(class_one, axis=0)).T,
                   (class_one - np.mean(class_one, axis=0)))
    # total within-class scatter matrix
    S_w = S_zero + S_one
    return S_w


def ComputeSB(X):
    # overall mean
    overall_mean = np.mean(X[:, :2], axis=0)
    # means of each class
    mean_diff = ComputeMeanDiff(X)
    mean_zero = overall_mean - mean_diff / 2
    mean_one = overall_mean + mean_diff / 2
    # between-class scatter matrix
    S_b = np.outer(mean_diff, mean_diff)
    return S_b


def GetLDAProjectionVector(X):
    # Get SW and SB
    S_w = ComputeSW(X)
    S_b = ComputeSB(X)
    # inverse of Sw
    S_w_inv = np.linalg.inv(S_w)
    # Sw^-1 dot Sb
    S_w_inv_Sb = np.dot(S_w_inv, S_b)
    # Compute the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(S_w_inv_Sb)
    # Get the eigenvector corresponding to the highest eigenvalue
    max_index = np.argmax(eigenvalues)
    w = eigenvectors[:, max_index]
    return w


def project(x, y, w):
    # Project the point (x, y) using the LDA projection vector w
    point = np.array([x, y])
    projection = np.dot(point, w)
    return projection


#########################################################
###################Helper Code###########################
#########################################################


X = np.empty((0, 3))
with open('data.csv', mode='r')as file:
    csvFile = csv.reader(file)
    for sample in csvFile:
        X = np.vstack((X, [float(num) for num in sample]))

print(X)
print(X.shape)
# X Contains m samples each of formate (x,y) and class label 0.0 or 1.0

opt = int(input("Input your option (1-5): "))

match opt:
    case 1:
        meanDiff = ComputeMeanDiff(X)
        print(meanDiff)
    case 2:
        SW = ComputeSW(X)
        print(SW)
    case 3:
        SB = ComputeSB(X)
        print(SB)
    case 4:
        w = GetLDAProjectionVector(X)
        print(w)
    case 5:
        x = float(input("Input x dimension of a 2-dimensional point :"))
        y = float(input("Input y dimension of a 2-dimensional point:"))
        w = GetLDAProjectionVector(X)
        projection = project(x, y, w)
        print(projection)
