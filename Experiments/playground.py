import matplotlib.pyplot as plt
import pandas as pd
from surprise import SVD, SVDpp, KNNBasic, KNNWithMeans, KNNWithZScore, NMF, SlopeOne, Dataset, Reader, \
    NormalPredictor, BaselineOnly, CoClustering, KNNBaseline
from surprise.model_selection import train_test_split, GridSearchCV
import recmetrics
import pickle

# Display maximum columns
from Algorithms.KFN import KFN
from Algorithms.KFN2 import KFN2

pd.set_option("display.max_columns", None)

# Read CSV
ratings = pd.read_csv('../Datasets/ml-20m/ratings.csv')
ratings.reset_index(drop=True, inplace=True)

# Find users who have rated more than 1000 movies
users = ratings["userId"].value_counts()
users = users[users > 2000].index.tolist()

# Filter ratings according to corresponding users
ratings = ratings.query('userId in @users')
rated_movies = ratings["movieId"].tolist()

# Find corresponding rated books
movies = pd.read_csv('../Datasets/ml-20m/movies.csv')
movies = movies.query('movieId in @rated_movies')
movies.set_index("movieId", inplace=True, drop=True)

# Preprocessing movie genre
movies = movies["genres"].str.split("|", expand=True)
movies.reset_index(inplace=True)

# Reshapes the data frame
movies = pd.melt(movies, id_vars='movieId', value_vars=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Remove duplicate rows
movies.drop_duplicates("movieId", inplace=True)
movies.set_index('movieId', inplace=True)

# Reshaping into dummy code
movies = pd.get_dummies(movies.value)

# Getting unique movieIDs
catalog = ratings.movieId.unique().tolist()

# Recommender system

# Preparing data for system
print("Performing train-test split...")
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainSet, testSet = train_test_split(data, test_size=0.2)
print("Train-test split complete!")

kfn2Algo = KFN2()
kfn2Algo.fit(trainSet)

knnAlgo = KNNBasic()
knnAlgo.fit(trainSet)

test = kfn2Algo.test(testSet)
test = pd.DataFrame(test)
print(test.head())
