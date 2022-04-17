from surprise import AlgoBase, PredictionImpossible
from surprise import Dataset
from surprise.model_selection import train_test_split, cross_validate
import numpy as np


class diversityAlgorithm(AlgoBase):
    def __init__(self, sim_options={}, bsl_options={}):
        AlgoBase.__init__(self, sim_options=sim_options,
                          bsl_options=bsl_options)
