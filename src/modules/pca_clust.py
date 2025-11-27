#Imports 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import umap
import matplotlib.pyplot as plt

#%$ PCA function
def our_pca(latent_damaged, latent_original, n_components=300):
    pca = PCA(n_components=n_components)

    smaller_latent_damaged = pca.fit_transform(latent_damaged)
    smaller_latent_original = pca.transform(latent_original)

    return smaller_latent_damaged, smaller_latent_original

#%% Clustering function
def clustering(latent_damaged, latent_original, n_clusters=10, n_iters=100):
    all_data = torch.cat((latent_damaged, latent_original), dim=0)
    N, D = all_data.shape

    indices = torch.randperm(N)[:n_clusters]
    centroids = all_data[indices]

    for _ in range(n_iters):
        distances = torch.cdist(all_data, centroids)
        labels = torch.argmin(distances, dim=1)      

        for k in range(n_clusters):
            if (labels == k).any():
                centroids[k] = all_data[labels == k].mean(dim=0)

    damaged_labels = labels[:len(latent_damaged)]
    original_labels = labels[len(latent_damaged):]

    return damaged_labels, original_labels

#%% UMAP visualization function
def display_umap(latent_original, original_labels, n_clusters=10):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(latent_original.cpu().numpy())

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=original_labels.cpu().numpy(), cmap='Spectral', s=5)
    plt.colorbar(boundaries=np.arange(n_clusters+1)-0.5).set_ticks(np.arange(n_clusters))
    plt.title('UMAP projection of the original latent space', fontsize=15)
    plt.show()
