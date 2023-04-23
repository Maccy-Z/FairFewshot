import torch
import pickle
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np

def main():
    with open("/mnt/storage_ssd/FairFewshot/saves/aaab_1/weights", "rb") as f:
        all_weights = pickle.load(f)

    wanted_weights = []
    labels = []
    for ds_name, weights in all_weights.items():
        print(ds_name)
        # weight_list: shape[0] = [BS, num_layers, tensor[4]]
        #              shape[1] = [BS, tensor[1]]
        wanted_weights.append(weights[1][0].flatten())
        labels.append(ds_name)

    preds = torch.stack(wanted_weights)
    labels = np.array(labels)

    # Compute t-SNE embedding
    tsne = TSNE(n_components=2, random_state=42, perplexity=81)
    embedded_data = tsne.fit_transform(preds)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(embedded_data)
    print(cluster_labels)
    # Print out the labels of the items in each cluster
    for i in range(kmeans.n_clusters):
        cluster_items = labels[(cluster_labels == i)]
        print(f"Cluster {i}: {cluster_items}")
        print()

    # Plot the embedded data
    color_map = plt.colormaps["tab10"]
    plt.figure(figsize=(16, 16))
    plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=color_map(cluster_labels))
    for i, label in enumerate(labels):
        plt.annotate(label, (embedded_data[i, 0], embedded_data[i, 1]), fontsize=18)

    plt.show()


if __name__ == "__main__":
    main()


