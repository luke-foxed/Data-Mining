# Lab 04
# Luke Fox - 20076173

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import euclidean_distances, calinski_harabasz_score
from sklearn.cluster import KMeans, AgglomerativeClustering


def main():
    # --- STEP 1 --- #
    iris = datasets.load_iris()
    x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])

    # --- STEP 2 --- #
    cluster_list = create_clusters(x)

    # --- STEP 3 --- #
    plot_within_cluster_difference(cluster_list, x)

    # --- STEP 4 --- #
    plot_between_cluster_difference(cluster_list, x)

    # --- STEP 5 --- #
    plot_calinski_herbasz_index(cluster_list, "K-Means", x)

    # --- STEP 6 --- #
    find_natural_clustering_arrangement(cluster_list, x)

    # --- STEP 7 --- #
    hierarchical_cluster_list = find_arrangement_of_data_points(x)
    plot_calinski_herbasz_index(hierarchical_cluster_list, "Hierarchical", x)

    # --- STEP 8 --- #
    # Looking at the plot from step 7 above, we can see that the natural arrangement is again 3. This is because
    # hierarchical clustering creates clusters with predominant ordering from top to bottom. Looking at the graph,
    # a cluster size of 3 has the highest CH index and is most predominant


def create_clusters(data):
    cluster_list = []
    # build clusters ranging from size 2-10
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i).fit(data)
        cluster_list.append(kmeans)
    return cluster_list


def plot_within_cluster_difference(cluster_list, data):
    sum_of_squares = []
    no_of_clusters = []

    # retrieve inertia (within-cluster sum of squares) and cluster number
    for i in cluster_list:
        sum_of_squares.append(i.inertia_)
        no_of_clusters.append(i.n_clusters)

    plt.scatter(no_of_clusters, sum_of_squares)
    plt.title("Within Clusters")
    plt.xlabel('No. of Clusters')
    plt.ylabel('Sum of Squares')
    plt.show()


def plot_between_cluster_difference(cluster_list, data):
    distances_between = []
    no_of_clusters = []

    # to find center, set cluster size to only 1
    center = KMeans(n_clusters=1).fit(data)

    for i in cluster_list:
        distance = 0
        values, count = np.unique(i.labels_, return_counts=True)

        # find dot product of cluster count and the square of cluster 2d array and cluster centers
        for cluster, count in zip(i.cluster_centers_, count):
            distance += np.dot(count, np.square(euclidean_distances([cluster], center.cluster_centers_)))

        # append distances between each cluster to array
        distances_between.append((distance))
        no_of_clusters.append(i.n_clusters)

    plt.scatter(no_of_clusters, distances_between)
    plt.title("Between Clusters")
    plt.xlabel('No. of Clusters')
    plt.ylabel('Distance Between')
    plt.show()


def plot_calinski_herbasz_index(cluster_list, cluster_type, data):
    ch_index = []
    no_of_clusters = []

    # similar process as before, but CH-score requires array of samples and array of labels
    for i in cluster_list:
        labels = i.labels_
        ch_index.append(calinski_harabasz_score(data, labels))
        no_of_clusters.append(i.n_clusters)

    plt.scatter(no_of_clusters, ch_index)
    plt.title("Calinksi Harabasz Plot ({0})".format(cluster_type))
    plt.xlabel('No. of Clusters')
    plt.ylabel('CH Index')
    plt.show()

def find_natural_clustering_arrangement(cluster_list, data):
    error = []

    # use elbow method to determine most optimal number of clusters
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i).fit(data)
        kmeans.fit(data)
        error.append(kmeans.inertia_)

    plt.plot(range(1, 11), error)
    plt.title('Cluster Arrangement (Elbow method)')
    plt.xlabel('No. of clusters')
    plt.ylabel('Error')
    plt.show()

    # as shown from the plot, the elbow is formed at between 2-4 clusters, therefore 3 is selected as the natural-
    # clustering arrangment. the reason for this is that the iris dataset contains 3 classes of 50 instances, so the
    # clustering is naturally arranged into these 3 classses

    kmeans = KMeans(n_clusters=3).fit(data)


def find_arrangement_of_data_points(data):
    cluster_list = []

    # build a hierarchical cluster list using AgglomerativeClustering, which will again be used with CH-score
    for i in range(2,11):
        hierarchical_clustering = AgglomerativeClustering(n_clusters=i).fit(data)
        cluster_list.append(hierarchical_clustering)
    return cluster_list


if __name__ == '__main__':
    main()
