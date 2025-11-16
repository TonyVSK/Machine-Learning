import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. Criar um dataset artificial para exemplificar o funcionamento
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# 2. Definir e treinar o modelo de clustering
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# 3. Plotar os clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Centroides')
plt.title("Agrupamentos gerados pelo KMeans")
plt.legend()
plt.show()