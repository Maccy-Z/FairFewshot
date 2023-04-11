import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import torch

from dataset2vec import Dataset2Vec, ResBlock
from d2v_dataset import Dataloader


dl = Dataloader(min_ds_len=500)
dataset = [x for x in dl.ds]
print(f'{len(dataset) = }')
labels = np.array([str(ds)[4:] for ds in dataset])

data = []
for ds in dataset:
    xs = ds.get_data(nsamples=50)
    data.append(xs)

model = Dataset2Vec(64, 64, [7, 5, 7])
state = torch.load("./dataset2vec/model")["state_dict"]
model.load_state_dict(state)

preds = model(data).detach().numpy()


# Compute t-SNE embedding
tsne = TSNE(n_components=2, random_state=42, perplexity=32)
embedded_data = tsne.fit_transform(preds)

# Perform K-means clustering
kmeans = KMeans(n_clusters=8, random_state=42)
cluster_labels = kmeans.fit_predict(embedded_data)

# Print out the labels of the items in each cluster
print(cluster_labels)
print(np.argwhere((cluster_labels == 0)).squeeze())
for i in range(kmeans.n_clusters):
    cluster_items = labels[(cluster_labels == i)]
    print(f"Cluster {i+1}: {cluster_items}")

print(embedded_data.shape)


# Plot the embedded data
plt.figure(figsize=(16, 16))
plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
for i, label in enumerate(labels):
    plt.annotate(label, (embedded_data[i, 0], embedded_data[i, 1]), fontsize=18)

plt.show()
