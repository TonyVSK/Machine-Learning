import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans

# 1. Gerar dataset em forma de duas luas
X, y = make_moons(n_samples=300, noise=0.07, random_state=42)

# 2. Aplicar K-Means com 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# 3. Plotar os clusters definidos pelo K-Means
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=50)
plt.title("Separação feita pelo K-Means")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()