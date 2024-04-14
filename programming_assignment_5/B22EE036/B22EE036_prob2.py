# %% [markdown]
# *   AUTHOR: K. K. N. SHYAM SATHVIK
# *   ROLL. NO: B22EE036
# 
# 

# %% [markdown]
# 
# 
# ---
# 
# 

# %% [markdown]
# # <b> Programming Assignment-5 (Lab-9&10)

# %% [markdown]
# ## <b>CSL2050 - Pattern Recognition and Machine Learning
# 

# %% [markdown]
# ### <b> TASK 2 : Support Vector Machine

# %%
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Imports

# %%
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ### Task Specific Imports

# %%
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_moons

# %% [markdown]
# ### Task-1
# <li>(a): Load the Iris dataset using the following code:

# %%
iris = datasets.load_iris(as_frame=True)

# %%
print(dir(iris))

# %%
print(iris.target)

# %%
print(iris.data)

# %%
display(iris.frame)

# %%
print(iris.target_names)

# %%
df = iris.frame

# %%
# dropping virginica
df = df[(df['target'] == 0) | (df['target'] == 1)]  # setosa and versicolor
X = df[['petal length (cm)', 'petal width (cm)']]
y = df['target']

# %%
display(df)

# %% [markdown]
# #### Data Processing

# %%
# normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# %% [markdown]
# #### Data Visualization

# %%
sns.set_style("darkgrid")

# petal length vs petal width
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='target', style='target', palette='bright', markers=['o', 's'])
plt.title('Scatter Plot of Petal Length vs Petal Width')
plt.show()

# %%
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['petal length (cm)'], kde=True, color='pink')
plt.title('Histogram of Petal Length')
plt.subplot(1, 2, 2)
sns.histplot(df['petal width (cm)'], kde=True, color='olive')
plt.title('Histogram of Petal Width')
plt.show()

# %%
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, y='petal length (cm)', x='target', palette='pastel', color="green")
plt.title('Box Plot of Petal Length by Species')
plt.subplot(1, 2, 2)
sns.boxplot(data=df, y='petal width (cm)', x='target', palette='pastel', color = "blue")
plt.title('Box Plot of Petal Width by Species')
plt.show()

# %%
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.violinplot(data=df, y='petal length (cm)', x='target', palette='pastel')
plt.title('Violin Plot of Petal Length by Species')
plt.subplot(1, 2, 2)
sns.violinplot(data=df, y='petal width (cm)', x='target', palette='pastel')
plt.title('Violin Plot of Petal Width by Species')
plt.show()

# %%
sns.pairplot(df, hue='target', markers=['o', 's'], palette='bright', diag_kind='kde')
plt.suptitle('Pair Plot of Iris Features', y=1.02)
plt.show()

# %% [markdown]
# ### Task 1(b): Train LinearSVC and Plot Decision Boundaries

# %%
# linear svc
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

# %%
# plot boundaries
# def plot_decision_boundaries(X, y, model, title="Decision Boundary"):
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
#                          np.arange(y_min, y_max, 0.02))
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)

#     plt.contourf(xx, yy, Z, alpha=0.8)
#     plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='g')
#     plt.xlabel('Petal Length')
#     plt.ylabel('Petal Width')
#     plt.title(title)
#     plt.show()

def plot_decision_boundaries(X, y, model, title="Decision Boundary"):
    # meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # prediction over grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(6, 4))

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, style=y, palette='bright', markers=['s', 'o'], edgecolor='w')
    plt.contourf(xx, yy, Z, alpha=0.5, levels=np.linspace(Z.min(), Z.max(), 3), cmap='coolwarm')

    # labelling
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title(title)
    plt.show()

# %%
# train data plot
plot_decision_boundaries(X_train, y_train, svc, "Decision Boundary on Training Data")

# %%
# test data plot
plot_decision_boundaries(X_test, y_test, svc, "Decision Boundary on Test Data")

# %% [markdown]
# #### Task-2 (a):
# Generate a synthetic dataset using the make moons() function from scikit-learn Take around 500 data points, and add 5% noise (misclassifications) to the dataset.

# %%
# generating data
X, y = make_moons(n_samples=500, noise=0.05, random_state=42)

# %%
pprint.pprint(X)

# %%
pprint.pprint(y)

# %%
# normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
pprint.pprint(X_scaled)

# %% [markdown]
# #### Data Visualization

# %%
plt.figure(figsize=(6, 4))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, style=y, palette='bright', markers=['s', 'o'], edgecolor='w')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title("Scatter Plot of Moons Dataset")
plt.show()

# %% [markdown]
# #### Task-2 (b):
# Implement SVM models with three different kernels: Linear, Polynomial, and RBF. Plot the decision boundaries for each kernel on the synthetic dataset. Analyze and comment on the differences in decision boundaries produced by these kernels. (5 pts)

# %%
# different kernels
kernels = ['linear', 'poly', 'rbf']
models = {}

# %% [markdown]
# #### Helper Code for Decision Boundaries

# %%
# function to plot boundaries
def plot_decision_boundaries_moons(X, y, model, title):
    plt.figure(figsize=(6, 4))
    xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                         np.linspace(-3, 3, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, style=y, palette='bright', markers=['s', 'o'])
    plt.contourf(xx, yy, Z, alpha=0.5, levels=np.linspace(Z.min(), Z.max(), 3), cmap='coolwarm')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()

# %%
# training loop
for kernel in kernels:
    model = SVC(kernel=kernel, random_state=42)
    model.fit(X_scaled, y)
    models[kernel] = model
    plot_decision_boundaries_moons(X_scaled, y, model, f"Decision Boundary with {kernel.capitalize()} Kernel")

# %% [markdown]
# #### Task-2 (c):
# Focus on the RBF kernel SVM model. Perform hyperparameter tuning to find the best values of gamma and C for this model. You can use techniques like grid search or random search. (2 pts)

# %%
# parameters to tune the model on
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1, 10, 100]
}

# %%
# performing grid search
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_scaled, y)

# %%
# getting the best parameters
best_params = grid_search.best_params_
best_svc = grid_search.best_estimator_
print("Best parameters:", best_params)

# %% [markdown]
# #### Task-2 (d):
# Plot the decision boundary for the RBF kernel SVM with the best Hyperparameters. Explain the impact of the selected gamma and C values on the model’s performance and decision boundary. Note: Ensure to complete each task thoroughly and document your findings in the lab report. (2 pts)

# %%
plot_decision_boundaries_moons(X_scaled, y, best_svc, "Decision Boundary with Best RBF Kernel")

# %% [markdown]
# 
# 
# ---
# 
# 


