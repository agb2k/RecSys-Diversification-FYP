import heapq
from surprise import PredictionImpossible
from surprise.prediction_algorithms.knns import SymmetricAlgo


class KNNOFNFN(SymmetricAlgo):

    def __init__(self, k=5, k2=5, k3=5, min_k=1, sim_options={}, verbose=True, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, verbose=verbose,
                               **kwargs)
        self.k = k
        self.k2 = k2
        self.k3 = k3
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities()

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        x, y = self.switch(u, i)

        # Find neighbours and similarity
        neighbors = [(self.sim[x, x2], r, x2) for (x2, r) in self.yr[y]]

        # Arrange from furthest to closest and chose k of the furthest
        kfn = heapq.nsmallest(self.k, neighbors, key=lambda t: t[0])

        # Find the neighbours of the KFN
        neighborsOfKFN = [(self.sim[x2, x3], r, x3) for (sim, r, x2) in kfn for (x3, r) in self.yr[y]]

        # Arrange from furthest to closest and chose k of the furthest
        kfnofn = heapq.nsmallest(self.k, neighborsOfKFN, key=lambda t: t[0])

        # Find neighbours and similarity
        Neighbors2 = [(self.sim[x, x3], r, x3) for (sim, r, x3) in kfnofn]

        # Choose k of the closest neighbours
        k_nearest_neighbors_of_furthest_neighbors = heapq.nlargest(self.k, Neighbors2,
                                                                   key=lambda t: t[0])  # Furthest Neighbours

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (sim, r, x3) in k_nearest_neighbors_of_furthest_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details
