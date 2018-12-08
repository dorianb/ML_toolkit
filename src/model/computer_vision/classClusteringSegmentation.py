from sklearn.cluster import KMeans


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

        if algorithm == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, max_iter=300, tol=0.0001)
        else:
            raise Exception("Algorithm is not yet implemented")

    def fit_predict(self, X):
        """
        Compute parameters of the model and predict the cluster of each pixel of an image.

        Args:
            X: an ndarray representing an image (x, y, color_dimension)

        Returns:
            an ndarray representing the image (x, y, cluster)

        """
        if self.algorithm == 'kmeans':
            return self.model.fit_predict(X)