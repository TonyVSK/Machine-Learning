import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import numpy as np

# 1. Gerar dataset em forma de duas luas
X, y = make_moons(n_samples=300, noise=0.07, random_state=42)

# 2. Aplicar DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

# 3. Plotar os clusters definidos pelo DBSCAN
plt.figure(figsize=(8,6))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Pontos de ruído (em preto)
        col = [0, 0, 0, 1]
        label_name = 'Ruído'
    else:
        label_name = f'Cluster {k+1}'

    mask = (labels == k)
    plt.scatter(X[mask, 0], X[mask, 1], c=[col], label=label_name, s=50)

plt.title("Two Moons com DBSCAN (ε=0.2, MinPts=5)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

# 4. Imprimir informações sobre os clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f'Número de clusters encontrados: {n_clusters}')
print(f'Número de pontos de ruído: {n_noise}')