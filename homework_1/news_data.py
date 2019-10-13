print(__doc__)

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np
from time import time

categories = ['alt.atheism'
    , 'comp.graphics'
    , 'comp.os.ms-windows.misc'
    , 'comp.sys.ibm.pc.hardware'
    , 'comp.sys.mac.hardware'
    # , 'comp.windows.x'
    # , 'misc.forsale'
    # , 'rec.autos'
    # , 'rec.motorcycles'
    # , 'rec.sport.baseball'
    # , 'rec.sport.hockey'
    # , 'sci.crypt'
    # , 'sci.electronics'
    # , 'sci.med'
    # , 'sci.space'
    # , 'soc.religion.christian'
    # , 'talk.politics.guns'
    # , 'talk.politics.mideast'
    # , 'talk.politics.misc'
    # , 'talk.religion.misc'
              ]
# categories = None
dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)
labels_true = dataset.target
true_k = np.unique(labels_true).shape[0]
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                             min_df=2, stop_words='english',
                             use_idf=True)
X = vectorizer.fit_transform(dataset.data)
svd = TruncatedSVD(100)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)
n_news = len(np.unique(labels_true))


def bench_clustring(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    # print(estimator.labels_)
    print('%-28s\t%.2fs\t%.3f\t%.3f\t%.3f\t'
          % (name, (time() - t0),
             metrics.homogeneity_score(labels_true, estimator.labels_),
             metrics.completeness_score(labels_true, estimator.labels_),
             metrics.normalized_mutual_info_score(labels_true, estimator.labels_,
                                                  average_method='arithmetic')))


if __name__ == '__main__':
    print(82 * '_')
    print('init\t\t\t\t\t\t\ttime\thomo\tcompl\tNMI')
    # bench_clustring(KMeans(n_clusters=true_k, init='k-means++', max_iter=100,
    #                        n_init=1, verbose=0),
    #                 name="k-means++", data=X)
    #
    # bench_clustring(AffinityPropagation(),
    #                 name="Affinity propagation", data=X)

    # wait to modify the args
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    bench_clustring(MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True,
                              n_jobs=None, seeds=None), name="Mean-shift", data=X)
    #
    # bench_clustring(SpectralClustering(affinity="nearest_neighbors"),
    #                 name="Spectral Clustering", data=X)
    #
    # bench_clustring(AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
    #                                         connectivity=None, distance_threshold=None,
    #                                         linkage='ward', memory=None, n_clusters=100,
    #                                         pooling_func='deprecated'),
    #                 name="Ward Hierarchical Clustering", data=X)
    #
    # bench_clustring(AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
    #                                         connectivity=None, distance_threshold=None,
    #                                         linkage='single', memory=None, n_clusters=512,
    #                                         pooling_func='deprecated'),
    #                 name="Agglomerative clustering", data=X)
    #
    # bench_clustring(DBSCAN(eps=0.005, min_samples=2),
    #                 name="DBSCAN", data=X)
    #
    # gm = GaussianMixture(n_components=50)
    # labels = gm.fit(X).predict(X)
    # t0 = time()
    # print('%-30s\t%.2fs\t%.3f\t%.3f\t%.3f\t'
    #       % ("Gaussian mixture", (time() - t0),
    #          metrics.homogeneity_score(labels_true, labels),
    #          metrics.completeness_score(labels_true, labels),
    #          metrics.normalized_mutual_info_score(labels_true, labels,
    #                                               average_method='arithmetic')))

    print(82 * '_')
