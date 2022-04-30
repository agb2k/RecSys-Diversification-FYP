from surprise import AlgoBase, PredictionImpossible

# Initial implementation of KFN, replaced by KFN2
class KFN(AlgoBase):

    def __init__(self, sim_options={}, bsl_options={}):

        AlgoBase.__init__(self, sim_options=sim_options,
                          bsl_options=bsl_options)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        # Compute baselines and similarities
        self.bu, self.bi = self.compute_baselines()
        self.sim = self.compute_similarities()

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        # Compute similarities between u and v, where v describes all other
        # users that have also rated item i.
        neighbors = [(v, self.sim[u, v]) for (v, r) in self.trainset.ir[i]]

        # Sort these neighbors by similarity
        neighbors_reverse = sorted(neighbors, key=lambda x: x[1], reverse=False)

        print('The 3 furthest neighbors of user', str(u), 'are:')
        for v, sim_uv in neighbors_reverse[:3]:
            print('user {0:} with sim {1:1.2f}'.format(v, sim_uv))

        sum_sim = sum_ratings = 0
        for v, sim_uv in neighbors_reverse:
            if sim_uv:
                sum_sim += sim_uv
                sum_ratings += sim_uv * v

        est = sum_ratings / sum_sim
        return est
