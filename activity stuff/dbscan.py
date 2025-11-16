import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# reproducibilidade
np.random.seed(42)

# 1. dados de pessoas numa cidade - terá 3 clusters com densidades diferentes e ruído

centers = [[2, 2], [6, 6], [2, 8]]  # Centros dos "bairros"
X, _ = make_blobs(n_samples=300, centers=centers, cluster_std=[0.6, 0.8, 0.5], random_state=42)

# Adicionar ruído em posições aleaatórias
noise = np.random.uniform(low=0, high=10, size=(50, 2))
X = np.vstack([X, noise])

# 2. Aplicar - DBSCAN eps (ε) = 0.5 (raio da vizinhança), min_samples (MinPts) = 5
dbscan = DBSCAN(eps=0.5, min_samples=20)
labels = dbscan.fit_predict(X)

# 3. Visualizar os resultados; ruído será marcado como -1 em preto
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(8, 6))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Pontos de ruído 
        col = [0, 0, 0, 1]
        label_name = 'Ruído'
    else:
        label_name = f'Cluster {k+1}'

    # Selecionar pontos do cluster atual
    mask = (labels == k)
    plt.scatter(X[mask, 0], X[mask, 1], c=[col], label=label_name, s=50)

plt.title('Pessoas em cidade\n(ε=0.5, MinPts=20)')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.legend()
plt.grid(True)
plt.show()

# 4. exibir informações sobre os clusters no terminal

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f'Número de clusters encontrados: {n_clusters}')
print(f'Número de pontos de ruído: {n_noise}')