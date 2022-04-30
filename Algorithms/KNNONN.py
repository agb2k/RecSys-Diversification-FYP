import heapq
from surprise import PredictionImpossible
from surprise.prediction_algorithms.knns import SymmetricAlgo


class KNNONN(SymmetricAlgo):

    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, verbose=verbose,
                               **kwargs)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities()

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        x, y = self.switch(u, i)

        neighbors = [(self.sim[x, x2], r, x2) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])  # Nearest Neighbours

        neighborsOfkNeighbors = [(self.sim[x2, x3], r, x3) for (sim, r, x2) in k_neighbors for (x3, r) in self.yr[y]]


        similarNeighbors = [(self.sim[x, x3], r, x3) for (sim, r, x3) in neighborsOfkNeighbors]

        k_nearest_neighbors_of_furthest_neighbors = heapq.nlargest(self.k, similarNeighbors,
                                                                   key=lambda t: t[0])  # Furthest Neighbours

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (sim, r, x3) in k_nearest_neighbors_of_furthest_neighbors:
            sum_sim += sim
            sum_ratings += sim * r
            actual_k += 1

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details