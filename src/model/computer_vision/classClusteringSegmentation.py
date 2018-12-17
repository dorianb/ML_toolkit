from sklearn.feature_extraction.image import img_to_graph
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering, spectral_clustering, AffinityPropagation
from sklearn.mixture import GaussianMixture


class ClusteringSegmentation:

    def __init__(self, algorithm, n_clusters):
        """
        Initialize a clustering segmentation object.

        Args:
            algorithm: the algorithm to use for clustering segmentation (kmeans and em)
            n_clusters: the number of clusters
        """
        self.algorithm = algorithm.lower()
        self.n_clusters = n_clusters

        if self.algorithm == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters,
                                max_iter=300, tol=0.0001)
        elif self.algorithm == 'gmm':
            self.model = GaussianMixture(n_components=self.n_clusters,
                                         covariance_type='full',
                                         tol=0.0001, reg_covar=1e-06,
                                         max_iter=300)
        elif self.algorithm == "affinity":
            self.model = AffinityPropagation(affinity='euclidean',
                                             convergence_iter=15, damping=0.5,
                                             max_iter=200, preference=None, verbose=False)
        elif self.algorithm == 'aglo':
            pass
        elif self.algorithm == 'spectral':
            pass
        else:
            raise Exception("Algorithm is not yet implemented")

    def fit(self, image):
        """
        Compute parameters of the model.

        Args:
            image: a ndarray representing an image (x, y, color_dimension)

        Returns:
            Nothing
        """
        f_dim = image.shape[-1] if len(image.shape) > 2 else 1
        X = image.reshape(-1, f_dim)

        if self.algorithm == 'aglo':
            connectivity = img_to_graph(image)
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters,
                                                 affinity='euclidean',
                                                 connectivity=connectivity,
                                                 compute_full_tree=False,
                                                 linkage='average')
        elif self.algorithm == 'spectral':
            return None

        self.model.fit(X)

    def predict(self, image):
        """
        Predict the cluster of each pixel of an image.

        Args:
            X: an ndarray representing an image (x, y, color_dim)

        Returns:
            an ndarray representing the image (x, y, cluster)

        """
        f_dim = image.shape[-1] if len(image.shape) > 2 else 1
        X = image.reshape(-1, f_dim)

        if self.algorithm == 'spectral':
            graph = img_to_graph(image)
            graph.data = np.exp(-graph.data / graph.data.std())

            X_clustered = spectral_clustering(
                graph, n_clusters=self.n_clusters, eigen_solver='arpack')
        else:
            X_clustered = self.model.predict(X)

        return X_clustered.reshape(image.shape[0], image.shape[1])

    def fit_predict(self, image):
        """
        Compute parameters of the model and predict the cluster of each pixel of an image.

        Args:
            image: an ndarray representing an image (x, y, color_dim)

        Returns:
            an ndarray representing the image (x, y, cluster)

        """
        self.fit(image)
        return self.predict(image)