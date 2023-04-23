# Makes t-sne plots with d2v dataset embeddings.
# Optionally, copies the datasets into folders according to k-nn of embeddings.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import torch
import shutil
import os

from dataset2vec import Dataset2Vec
from d2v_dataset import Dataloader

print(os.getcwd())

num_clusters = 6

dl = Dataloader(min_ds_len=50)

dataset = [x for x in dl.ds]
print(f'{len(dataset) = }')
labels = np.array([str(ds)[4:] for ds in dataset])

data = []
for ds in dataset:
    xs = ds.get_data(nsamples=100)
    data.append(xs)

load = torch.load("./dataset2vec/model_3")
state, params = load["state_dict"], load["params"]
model = Dataset2Vec(*params)
model.load_state_dict(state)
preds = model(data).detach().numpy()


# Compute t-SNE embedding
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embedded_data = tsne.fit_transform(preds)

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embedded_data)
# Print out the labels of the items in each cluster
for i in range(kmeans.n_clusters):
    cluster_items = labels[(cluster_labels == i)]
    print(f"Cluster {i}: {cluster_items}")
    print()

# Plot the embedded data
color_map = plt.colormaps["tab10"]
plt.figure(figsize=(32, 32))
plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=color_map(cluster_labels))
for i, label in enumerate(labels):
    plt.annotate(label, (embedded_data[i, 0], embedded_data[i, 1]), fontsize=18)

#plt.show()
plt.savefig('/Users/kasiakobalczyk/Fairfewshot/embeddings.pdf')

command = input("If you are happy with the clustering, type 'YES' clusters will be saved, deleting old clusters: \n")
if command != "YES":
    print("Nothing done. Exiting")

# # Empty dir
# print("Deleting all old files")
# path = f'./datasets/grouped_datasets/'
# for file_name in os.listdir(path):
#     file_path = os.path.join(path, file_name)
#     try:
#         if os.path.isfile(file_path):
#             os.remove(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#     except Exception as e:
#         print(f"Failed to delete {file_path}. Reason: {e}")


# datasets = np.array(dataset)
# for i in range(num_clusters):
#     mask = (cluster_labels == i)
#     selected_ds = labels[mask]

#     for ds in selected_ds:

#         dst_dir = f'./datasets/grouped_datasets/{i}/{ds}'
#         if os.path.exists(f'./datasets/grouped_datasets/{i}/{ds}'):
#             shutil.rmtree(dst_dir)

#         shutil.copytree(f'./datasets/data/{ds}', f'./datasets/grouped_datasets/{i}/{ds}')
    # if os.path.exists("./datasets/data/"):
    #     print("EXISTS")
    # else:
    #     exit(6)

# print(datasets)

