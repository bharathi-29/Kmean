import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Load the data
data = pd.read_csv("kmeansdata.csv")
df1 = pd.DataFrame(data)
print(df1)

# Extract features
f1 = df1['Distance_Feature'].values
f2 = df1['Speeding_Feature'].values
X = np.array(list(zip(f1, f2)))

# Initial plot of the dataset
plt.figure()
plt.xlim([0, 100])
plt.ylim([0, 50])
plt.title('Dataset')
plt.xlabel('Distance_Feature')
plt.ylabel('Speeding_Feature')
plt.scatter(f1, f2)
plt.show()

# Define colors and markers for clusters
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# Apply KMeans algorithm with 3 clusters
kmeans_model = KMeans(n_clusters=3, random_state=0).fit(X)

# Plot the clustered data points
plt.figure()
for i, label in enumerate(kmeans_model.labels_):
    plt.plot(f1[i], f2[i], color=colors[label], marker=markers[label], ls='None')

plt.xlim([0, 100])
plt.ylim([0, 50])
plt.title('KMeans Clustering')
plt.xlabel('Distance_Feature')
plt.ylabel('Speeding_Feature')
plt.show()
