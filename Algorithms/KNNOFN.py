import heapq
from surprise import PredictionImpossible
from surprise.prediction_algorithms.knns import SymmetricAlgo


# K-Nearest Neighbours of Furthest Neighbours
class KNNOFN(SymmetricAlgo):

    def __init__(self, k=50, k2=20, min_k=1, sim_options={}, verbose=True, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, verbose=verbose,
                               **kwargs)
        self.k = k
        self.k2 = k2
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities()

        return self

    def estimate(self, u, i):

        # Raise prediction impossible if user or item aren't in train set
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        # Switch between user-based or item-based
        x, y = self.switch(u, i)

        # Find neighbours and similarity
        neighbors = [(self.sim[x, x2], r, x2) for (x2, r) in self.yr[y]]

        # Arrange from furthest to closest and chose k of the furthest (nsmallest similarity)
        k_neighbors = heapq.nsmallest(self.k, neighbors, key=lambda t: t[0])

        # Find the neighbours of the k-furthest neighbours
        neighborsOfkNeighbors = [(self.sim[x2, x3], r, x3) for (sim, r, x2) in k_neighbors for (x3, r) in self.yr[y]]

        # Find neighbours and similarity
        similarNeighbors = [(self.sim[x, x3], r, x3) for (sim, r, x3) in neighborsOfkNeighbors]

        # Choose k of the neighbours of the k-Furthest neighbours with the greatest similarity to the initial point
        KNN_of_FN = heapq.nlargest(self.k2, similarNeighbors, key=lambda t: t[0])

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (sim, r, x3) in KNN_of_FN:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        # Raise prediction impossible exception
        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        # Estimated rating
        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details
