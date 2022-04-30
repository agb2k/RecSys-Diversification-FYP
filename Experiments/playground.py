import matplotlib.pyplot as plt
import pandas as pd
from surprise import SVD, SVDpp, KNNBasic, KNNWithMeans, KNNWithZScore, NMF, SlopeOne, Dataset, Reader, \
    NormalPredictor, BaselineOnly, CoClustering, KNNBaseline
from surprise.model_selection import train_test_split, GridSearchCV
import recmetrics
import pickle

# Display maximum columns
from Algorithms.KNNOFN import KNNOFN

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
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainSet, testSet = train_test_split(data, test_size=0.2, random_state=22092000)

algoList = []

knnon2Algo = KNNOFN()
knnon2Algo.fit(trainSet)
algoList.append(knnon2Algo)
for algo in algoList:
    test = algo.test(testSet)
    test = pd.DataFrame(test)
    test.drop("details", inplace=True, axis=1)
    test.columns = ['userId', 'movieId', 'actual', 'predictions']

    # Calculate mse and rmse
    mse = recmetrics.mse(test.actual, test.predictions)

    rmse = recmetrics.rmse(test.actual, test.predictions)

    # Create model (matrix of predicted values)
    algoModel = test.pivot_table(index='userId', columns='movieId', values='predictions').fillna(0)


    # Gets user predictions and returns it in a list form
    def get_users_predictions(user_id, n, model):
        recommended_items = pd.DataFrame(model.loc[user_id])
        recommended_items.columns = ["predicted_rating"]
        recommended_items = recommended_items.sort_values('predicted_rating', ascending=False)
        recommended_items = recommended_items.head(n)
        return recommended_items.index.tolist()


    test = test.copy().groupby('userId', as_index=False)['movieId'].agg({'actual': (lambda x: list(set(x)))})
    test = test.set_index("userId")

    # Make recommendations for all members in the test data
    recs = [] = []
    for user in test.index:
        predictions = get_users_predictions(user, 10, algoModel)
        recs.append(predictions)

    test['predictions'] = recs

    nov = ratings.movieId.value_counts()
    pop = dict(nov)

    # Calculate novelty and personalization
    novelty, mselfinfo_list = recmetrics.novelty(recs, pop, len(users), 10)

    personalization = recmetrics.personalization(recs)

    intraSim = recmetrics.intra_list_similarity(recs, movies)

    predCov = recmetrics.prediction_coverage(recs, catalog)
    catCov = recmetrics.catalog_coverage(recs, catalog, 100)

    print(f"\n\n<------------------------------------------------------------------>\n")
    print(f"Algorithm Configuration: k={algo.k};k2={algo.k2}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"Novelty: {novelty}")
    print(f"Personalization: {personalization}")
    print(f"Intra-list Similarity: {intraSim}")
    print(f"Prediction Coverage: {predCov}")
    print(f"Catalog Coverage: {catCov}")

    print(f"Novelty : RMSE:{novelty / rmse}")
    print(f"Personalization : RMSE:{personalization / rmse}")
    print(f"Intra-list Similarity : RMSE:{intraSim / rmse}")
    print(f"Prediction Coverage : RMSE: {predCov / rmse}")
    print(f"Catalog Coverage : RMSE: {catCov / rmse}")
    print(f"\n<------------------------------------------------------------------>\n")
