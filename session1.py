from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
inertia= []
k_values= range(1,8)
spending = np.array([10, 12, 11, 90, 95, 88, 50, 52, 48, 15, 13, 91])
x= spending.reshape(-1,1)
for k in k_values:
    km= KMeans(n_clusters=k , random_state=42)
    km.fit(x)
    inertia.append(km.inertia_) 

plt.plot(k_values, inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()

from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x)

score = silhouette_score(x, kmeans.labels_)
print(f"Silhouette Score: {score:.2f}")
