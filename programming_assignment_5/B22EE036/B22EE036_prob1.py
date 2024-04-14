# %% [markdown]
# # <b> Programming Assignment-5 (Lab-9&10)

# %% [markdown]
# ## <b>CSL2050 - Pattern Recognition and Machine Learning
# 

# %% [markdown]
# ### <b> TASK 1: Image Compression Using K-Means Clustering </b>
# 
# <li> (Image Compression using K-means) You are given an RGB image.
# <li> Consider each pixel value as a 3-dimensional feature. 
# <li> Use KMeans and represent each pixel by the centroid color.

# %%
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # Imports

# %%
import numpy as np
import matplotlib.pyplot as plt

# K-means
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

# image manipulations
from PIL import Image

# reading image from a url
import requests
from io import BytesIO

# %% [markdown]
# ### Task (a) : 
# <li> Implement a function – computeCentroid, that takes n 3-dimensional features and returns their mean. (2 pts)

# %%
def compute_centroid(features):
    # generalised for n-dimensional data
    return np.mean(features, axis=0)

# %% [markdown]
# ### Task (b) : 
# <li> Implement a function – mykmeans from scratch that takes data matrix X of size m×3 where m is the number of pixels in the image and the number
# of clusters k. 
# <li> It returns the cluster centers using the k-means algorithm. (3 pts)

# %% [markdown]
# #### Helper Functions:

# %%
def initialize_centroids(X, k):
    """Randomly initialize centroids from the dataset."""
    return X[np.random.choice(X.shape[0], k, replace=False), :]

# %%
def assign_clusters(X, centroids):
    """Assign each data point to the nearest centroid."""
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

# %%
def update_centroids(X, closest_centroids, k):
    """Compute new centroids as the mean of the points in each cluster."""
    return np.array([X[closest_centroids == i].mean(axis=0) for i in range(k)])

# %%
def check_convergence(centroids, new_centroids):
    """Check if centroids have changed; if not, we've converged."""
    return np.all(centroids == new_centroids)

# %% [markdown]
# #### K-Means

# %%
def mykmeans(X, k, max_iters=100):
    """K-Means Clustering Algorithm"""
    centroids = initialize_centroids(X, k)
    
    for _ in range(max_iters):
        closest_centroids = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, closest_centroids, k)
        if check_convergence(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return centroids


# %% [markdown]
# ### Task (c) : 
# <li> Use the centroids of k-means to represent the pixels of the image. 
# <li> Now, show compressed images for different values of k. (1 pts)

# %%
def compress_image(image_np, centroids, labels):
    # replace each pixel value with its centroid value
    compressed_image = centroids[labels]
    compressed_image = compressed_image.reshape(image_np.shape)
    return compressed_image.astype(np.uint8)

# %%
def display_compressed_images(image_path, ks=[2, 4, 8, 16]):
    image = Image.open(image_path)
    image_np = np.array(image)
    image_reshaped = image_np.reshape(-1, 3)
    
    plt.figure(figsize=(15, 10))
    
    for i, k in enumerate(ks):
        # call k-means
        centroids = mykmeans(image_reshaped, k)
        labels = pairwise_distances_argmin(image_reshaped, centroids)
        compressed_image = compress_image(image_np, centroids, labels)
        
        # different colors (centroids)
        plt.subplot(2, 2, i+1)
        plt.imshow(compressed_image)
        plt.title(f'Compressed Image with {k} Colors')
        plt.axis('off')
    plt.show()

# %%
# image_url = "https://raw.githubusercontent.com/boku13/pattern_recognition_and_machine_learning_labs/main/programming_assignment_5/test.png"
# image = load_image_from_url(image_url)
# image.show()

# %%
image_path = "test.png"

# %%
display_compressed_images(image_path)

# %% [markdown]
# ### Task (d) : 
# <li> Show the results of compressed images using the k-means implementa-
# tion of the sklearn library. What differences do you observe? (2 pts)

# %%
def compare_kmeans_methods(image_path, ks=[2, 4, 8, 16]):
    image = Image.open(image_path)
    image_np = np.array(image)
    image_reshaped = image_np.reshape(-1, 3)
    print(image_reshaped.shape)
    
    plt.figure(figsize=(15, 10))
    
    for i, k in enumerate(ks):
        # My k-means
        my_centroids = mykmeans(image_reshaped, k)
        my_labels = pairwise_distances_argmin(image_reshaped, my_centroids)
        my_compressed_image = compress_image(image_np, my_centroids[:, :3], my_labels)

        # Sklearn KMeans
        kmeans = KMeans(n_clusters=k, random_state=0)
        sklearn_labels = kmeans.fit_predict(image_reshaped)
        sklearn_centroids = kmeans.cluster_centers_
        sklearn_compressed_image = compress_image(image_np, sklearn_centroids, sklearn_labels)
        
        # Plotting
        plt.subplot(len(ks), 2, 2*i + 1)
        plt.imshow(my_compressed_image)
        plt.title(f'My KMeans (k = {k})')
        plt.axis('off')

        plt.subplot(len(ks), 2, 2*i + 2)
        plt.imshow(sklearn_compressed_image)
        plt.title(f'Sklearn KMeans (k = {k})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# %%
compare_kmeans_methods(image_path)

# %% [markdown]
# ### Task (e) : 
# <li> Spatial coherence: Incorporating spatial information helps maintain spa-tial coherence in the compressed image. 
# <li> Pixels that are nearby in the original image are more likely to be assigned to the same cluster, 
# preserving local structures and reducing artifacts like color bleeding or noise. 
# <li> How do you implement spatial coherence? Write the idea, implement it, and write down your observation. (2 pts))

# %%
def add_spatial_features(image_np, scale=0.1):
    m, n, _ = image_np.shape  
    X, Y = np.meshgrid(range(n), range(m))
    features = np.c_[image_np.reshape(-1, 3), scale * X.flatten(), scale * Y.flatten()]
    return features

# %%
def compare_kmeans_methods_with_spacial_coherence(image_path, ks=[2, 4, 8, 16]):
    image = Image.open(image_path)
    image_np = np.array(image)
    image_reshaped = image_np.reshape(-1, 3)
    print(image_reshaped.shape)
    spatial_features = add_spatial_features(image_np)
    print(spatial_features.shape)

    plt.figure(figsize=(15, 10))
    
    for i, k in enumerate(ks):
        # My k-means
        my_centroids = mykmeans(spatial_features, k)
        my_labels = pairwise_distances_argmin(spatial_features, my_centroids)
        my_compressed_image = compress_image(image_np, my_centroids[:, :3], my_labels)

        # My K-means without spacial coherence
        centroids_without_spacial_coherence = mykmeans(image_reshaped, k)
        labels_without_spacial_coherence = pairwise_distances_argmin(image_reshaped, 
                                                                     centroids_without_spacial_coherence)
        compressed_image_without_spacial_coherence = compress_image(image_np, 
                                                                    centroids_without_spacial_coherence[:, :3], 
                                                                    labels_without_spacial_coherence)
        
        # Plotting
        plt.subplot(len(ks), 2, 2*i + 1)
        plt.imshow(my_compressed_image)
        plt.title(f'My KMeans with Spacial Coherence (k = {k})')
        plt.axis('off')

        plt.subplot(len(ks), 2, 2*i + 2)
        plt.imshow(compressed_image_without_spacial_coherence)
        plt.title(f'My KMeans without Spacial Coherence (k = {k})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# %%
compare_kmeans_methods_with_spacial_coherence(image_path)

# %% [markdown]
# 
# 
# ---
# 
# 


