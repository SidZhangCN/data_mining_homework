print(__doc__)

from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

np.random.seed(42)
digits = load_digits()
data = scale(digits.data)
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels_true = digits.target
sample_size = 300


def bench_clustring(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-28s\t%.2fs\t%.3f\t%.3f\t%.3f\t'
          % (name, (time() - t0),
             metrics.homogeneity_score(labels_true, estimator.labels_),
             metrics.completeness_score(labels_true, estimator.labels_),
             metrics.normalized_mutual_info_score(labels_true, estimator.labels_,
                                                  average_method='arithmetic')))


if __name__ == '__main__':
    print("n_digits: %d, \t n_samples %d, \t n_features %d"
          % (n_digits, n_samples, n_features))
    print(82 * '_')
    print('init\t\t\t\t\t\t\ttime\thomo\tcompl\tNMI')

    bench_clustring(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
                  name="k-means++", data=data)

    bench_clustring(AffinityPropagation(preference=-50),
                               name="Affinity propagation", data=data)

    # wait to modify the args
    bandwidth = estimate_bandwidth(data, quantile=0.32, n_samples=100)
    bench_clustring(MeanShift(bandwidth=bandwidth, bin_seeding=True),
                    name="Mean-shift", data=data)

    bench_clustring(SpectralClustering(affinity="nearest_neighbors"),
                    name="Spectral Clustering", data=data)

    bench_clustring(AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                            connectivity=None, distance_threshold=None,
                            linkage='ward', memory=None, n_clusters=20,
                            pooling_func='deprecated'),
                            name="Ward Hierarchical Clustering", data=data)

    bench_clustring(AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                            connectivity=None, distance_threshold=None,
                            linkage='single', memory=None, n_clusters=512,
                            pooling_func='deprecated'),
                            name="Agglomerative clustering", data=data)

    bench_clustring(DBSCAN(eps=4, min_samples=0.5),
                  name="DBSCAN", data=data)

    gm = GaussianMixture(n_components=50)
    labels = gm.fit(data).predict(data)
    t0 = time()
    print('%-30s\t%.2fs\t%.3f\t%.3f\t%.3f\t'
          % ("Gaussian mixture", (time() - t0),
             metrics.homogeneity_score(labels_true, labels),
             metrics.completeness_score(labels_true, labels),
             metrics.normalized_mutual_info_score(labels_true, labels,
                                                  average_method='arithmetic')))

    print(82 * '_')
