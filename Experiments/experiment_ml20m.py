import matplotlib.pyplot as plt
import pandas as pd
from surprise import SVD, SVDpp, KNNBasic, KNNWithMeans, KNNWithZScore, NMF, SlopeOne, Dataset, Reader, \
    NormalPredictor, BaselineOnly, CoClustering, KNNBaseline, dump
from surprise.model_selection import train_test_split, GridSearchCV
from Algorithms.KFN2 import KFN2
import recmetrics
import pickle

trainBool = False

# Display maximum columns
pd.set_option("display.max_columns", None)

# Read CSV
ratings = pd.read_csv('../Datasets/ml-20m/ratings.csv')
ratings.reset_index(drop=True, inplace=True)

# Find users who have rated more than 1000 movies
users = ratings["userId"].value_counts()
users = users[users > 1000].index.tolist()

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

# Long Tail Plot for existing dataset ratings to show popularity of movies
fig = plt.figure(figsize=(15, 7))
recmetrics.long_tail_plot(df=ratings, item_id_column="movieId", interaction_type="movie ratings",
                          percentage=0.5, x_labels=False)

# Recommender system

# Preparing data for system
print("Performing train-test split...")
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainSet, testSet = train_test_split(data, test_size=0.2, random_state=22092000)
print("Train-test split complete!")

algoList = []

# Recommender System Algorithm, SVD
if not trainBool:
    print("Loading model...")
    svdAlgo = pickle.load(open('../Models/ML-20M Models/svd_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    svdAlgo = SVD()
    svdAlgo.fit(trainSet)
    pickle.dump(svdAlgo, open('../Models/ML-20M Models/svd_algo.sav', 'wb'))
    # dump.dump(svdAlgo, open('../Models/ML-20M Models/svd_algo.sav', 'wb'))
    print("Model trained!")
algoList.append(svdAlgo)

# Recommender System Algorithm, KNN
if not trainBool:
    print("Loading model...")
    knnAlgo = pickle.load(open('../Models/ML-20M Models/knn_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    knnAlgo = KNNBasic()
    knnAlgo.fit(trainSet)
    pickle.dump(knnAlgo, open('../Models/ML-20M Models/knn_algo.sav', 'wb'))
    print("Model trained!")
algoList.append(knnAlgo)

# Recommender System Algorithm, KNN with Means
if not trainBool:
    print("Loading model...")
    knnMeansAlgo = pickle.load(open('../Models/ML-20M Models/knn_means_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    knnMeansAlgo = KNNWithMeans()
    knnMeansAlgo.fit(trainSet)
    pickle.dump(knnMeansAlgo, open('../Models/ML-20M Models/knn_means_algo.sav', 'wb'))
    print("Model trained!")
algoList.append(knnMeansAlgo)

# Recommender System Algorithm, KNN with Z-Score
if not trainBool:
    print("Loading model...")
    knnZAlgo = pickle.load(open('../Models/ML-20M Models/knn_Z_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    knnZAlgo = KNNWithZScore()
    knnZAlgo.fit(trainSet)
    pickle.dump(knnZAlgo, open('../Models/ML-20M Models/knn_Z_algo.sav', 'wb'))
    print("Model trained!")
algoList.append(knnZAlgo)

# Recommender System Algorithm, KNN Baseline
if not trainBool:
    print("Loading model...")
    knnBaselineAlgo = pickle.load(open('../Models/ML-20M Models/knn_baseline_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    knnBaselineAlgo = KNNBaseline()
    knnBaselineAlgo.fit(trainSet)
    pickle.dump(knnBaselineAlgo, open('../Models/ML-20M Models/knn_baseline_algo.sav', 'wb'))
    print("Model trained!")
algoList.append(knnBaselineAlgo)

# Recommender System Algorithm, SVD plus plus
if not trainBool:
    print("Loading model...")
    svdPPAlgo = pickle.load(open('../Models/ML-20M Models/svd_pp_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    svdPPAlgo = SVDpp()
    svdPPAlgo.fit(trainSet)
    pickle.dump(svdPPAlgo, open('../Models/ML-20M Models/svd_pp_algo.sav', 'wb'))
    print("Model trained!")
algoList.append(svdPPAlgo)

# Recommender System Algorithm, Non-Negative Matrix Factorization
if not trainBool:
    print("Loading model...")
    nmfAlgo = pickle.load(open('../Models/ML-20M Models/nmf_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    nmfAlgo = NMF()
    nmfAlgo.fit(trainSet)
    pickle.dump(nmfAlgo, open('../Models/ML-20M Models/nmf_algo.sav', 'wb'))
    print("Model trained!")
algoList.append(nmfAlgo)

# # Recommender System Algorithm, Slope One
# if not trainBool:
#     print("Loading model...")
#     slopeAlgo = pickle.load(open('../Models/ML-20M Models/slope_algo.sav', 'rb'))
#     print("Model loaded!")
# else:
#     print("Training model...")
#     slopeAlgo = SlopeOne()
#     slopeAlgo.fit(trainSet)
#     pickle.dump(slopeAlgo, open('../Models/ML-20M Models/slope_algo.sav', 'wb'))
#     print("Model trained!")

# Recommender System Algorithm, Co-clustering
if not trainBool:
    print("Loading model...")
    ccAlgo = pickle.load(open('../Models/ML-20M Models/cc_algo', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    ccAlgo = CoClustering()
    ccAlgo.fit(trainSet)
    pickle.dump(ccAlgo, open('../Models/ML-20M Models/cc_algo', 'wb'))
    print("Model trained!")
algoList.append(ccAlgo)

# Recommender System Algorithm, Normal Predictor (Random)
if not trainBool:
    print("Loading model...")
    normalAlgo = pickle.load(open('../Models/ML-20M Models/normal_algo', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    normalAlgo = NormalPredictor()
    normalAlgo.fit(trainSet)
    pickle.dump(normalAlgo, open('../Models/ML-20M Models/normal_algo', 'wb'))
    print("Model trained!")
algoList.append(normalAlgo)

# Recommender System Algorithm, Baseline
if not trainBool:
    print("Loading model...")
    baselineAlgo = pickle.load(open('../Models/ML-20M Models/baseline_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    baselineAlgo = BaselineOnly()
    baselineAlgo.fit(trainSet)
    pickle.dump(baselineAlgo, open('../Models/ML-20M Models/baseline_algo.sav', 'wb'))
    print("Model trained!")
algoList.append(baselineAlgo)

# Recommender System Algorithm, K-Furthest Neighbours(V2)
if not trainBool:
    print("Loading model...")
    kfnAlgo = pickle.load(open('../Models/ML-20M Models/KFN.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    kfnAlgo = KFN2()
    kfnAlgo.fit(trainSet)
    pickle.dump(kfnAlgo, open('../Models/ML-20M Models/KFN.sav', 'wb'))
    print("Model trained!")
algoList.append(kfnAlgo)

test = None
mseList = []
rmseList = []
noveltyList = []
personList = []
intraSimList = []
predictionCoverageList = []
catalogCoverageList = []

# Testing algorithm with test data
for algo in algoList:
    test = algo.test(testSet)
    test = pd.DataFrame(test)
    test.drop("details", inplace=True, axis=1)
    test.columns = ['userId', 'movieId', 'actual', 'predictions']

    # Calculate mse and rmse
    print("Calculating MSE...")
    mse = recmetrics.mse(test.actual, test.predictions)
    print("Calculated MSE!")

    print("Calculating RMSE...")
    rmse = recmetrics.rmse(test.actual, test.predictions)
    print("Calculated RMSE!")

    # Create model (matrix of predicted values)
    print("Creating model...")
    algoModel = test.pivot_table(index='userId', columns='movieId', values='predictions').fillna(0)
    print("Created model!")


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
    print("Making recommendations for each member...")
    recs = [] = []
    for user in test.index:
        predictions = get_users_predictions(user, 10, algoModel)
        recs.append(predictions)
    print("Recommendations made for each member!")

    test['predictions'] = recs

    nov = ratings.movieId.value_counts()
    pop = dict(nov)

    # Calculate novelty and personalization
    print("Calculating Novelty...")
    novelty, mselfinfo_list = recmetrics.novelty(recs, pop, len(users), 10)
    print("Calculated Novelty!")

    print("Calculating Personalization...")
    personalization = recmetrics.personalization(recs)
    print("Calculated Personalization!")

    print("Calculating Intra-List Similarity...")
    intraSim = recmetrics.intra_list_similarity(recs, movies)
    print("Calculated Intra-List Similarity!")

    print("Calculating Prediction Coverage...")
    predCov = recmetrics.prediction_coverage(recs, catalog)
    print("Calculated Intra-List Similarity!")

    print("Calculating Coverage...")
    catCov = recmetrics.catalog_coverage(recs, catalog, 100)
    print("Calculated Intra-List Similarity!")

    mseList.append(mse)
    rmseList.append(rmse)
    noveltyList.append(novelty)
    personList.append(personalization)
    intraSimList.append(intraSim)
    predictionCoverageList.append(predCov)
    catalogCoverageList.append(catCov)
    print(algo)
    print(mse)
    print(rmse)
    print(novelty)
    print(personalization)
    print(intraSim)
    print(predCov)
    print(catCov)

# # <----------------------------- SVD-KNN Hybrid Algorithm ---------------------------------------------->
test1 = svdAlgo.test(testSet)
test1 = pd.DataFrame(test1)
test1.drop("details", inplace=True, axis=1)
test1.columns = ['userId', 'movieId', 'actual', 'SVD predictions']

test2 = knnAlgo.test(testSet)
test2 = pd.DataFrame(test2)
test2.drop("details", inplace=True, axis=1)
test2.columns = ['userId', 'movieId', 'actual', 'KNN predictions']
test2.drop(['userId', 'movieId', 'actual'], inplace=True, axis=1)

result = pd.concat([test1, test2], axis=1, join='inner')
col = result.loc[:, "SVD predictions": "KNN predictions"]
result['predictions'] = col.mean(axis=1)
result.drop(["SVD predictions", "KNN predictions"], inplace=True, axis=1)

print("Calculating MSE...")
mse = recmetrics.mse(result.actual, result.predictions)
print("Calculated MSE!")

print("Calculating RMSE...")
rmse = recmetrics.rmse(result.actual, result.predictions)
print("Calculated RMSE!")

# Create model (matrix of predicted values)
print("Creating model...")
algoModel = result.pivot_table(index='userId', columns='movieId', values='predictions').fillna(0)
print("Created model!")


# Gets user predictions and returns it in a list form
def get_users_predictions(user_id, n, model):
    recommended_items = pd.DataFrame(model.loc[user_id])
    recommended_items.columns = ["predicted_rating"]
    recommended_items = recommended_items.sort_values('predicted_rating', ascending=False)
    recommended_items = recommended_items.head(n)
    return recommended_items.index.tolist()


result = result.copy().groupby('userId', as_index=False)['movieId'].agg({'actual': (lambda x: list(set(x)))})
result = result.set_index("userId")

# Make recommendations for all members in the test data
print("Making recommendations for each member...")
recs = [] = []
for user in result.index:
    predictions = get_users_predictions(user, 10, algoModel)
    recs.append(predictions)
print("Recommendations made for each member!")

result['predictions'] = recs

nov = ratings.movieId.value_counts()
pop = dict(nov)

# Calculate novelty and personalization
print("Calculating Novelty...")
novelty, mselfinfo_list = recmetrics.novelty(recs, pop, len(users), 10)
print("Calculated Novelty!")

print("Calculating Personalization...")
personalization = recmetrics.personalization(recs)
print("Calculated Personalization!")

print("Calculating Intra-List Similarity...")
intraSim = recmetrics.intra_list_similarity(recs, movies)
print("Calculated Intra-List Similarity!")

print("Calculating Prediction Coverage...")
predCov = recmetrics.prediction_coverage(recs, catalog)
print("Calculated Intra-List Similarity!")

print("Calculating Coverage...")
catCov = recmetrics.catalog_coverage(recs, catalog, 100)
print("Calculated Intra-List Similarity!")

mseList.append(mse)
rmseList.append(rmse)
noveltyList.append(novelty)
personList.append(personalization)
intraSimList.append(intraSim)
predictionCoverageList.append(predCov)
catalogCoverageList.append(catCov)

# <-------------------- SVD-Normal Hybrid Algorithm ------------------------------->

test4 = normalAlgo.test(testSet)
test4 = pd.DataFrame(test4)
test4.drop("details", inplace=True, axis=1)
test4.columns = ['userId', 'movieId', 'actual', 'Normal predictions']
test4.drop(['userId', 'movieId', 'actual'], inplace=True, axis=1)
test4["Standardized"] = ((test4['Normal predictions'] / 5) - 0.5) * -1
print(test4.head())

result2 = pd.concat([test1, test4], axis=1, join='inner')
col = result2.loc[:, "SVD predictions": "Standardized"]
result2['predictions'] = result2["SVD predictions"] + result2["Standardized"]
print(result2.head())
result2.drop(["SVD predictions", "Standardized"], inplace=True, axis=1)

print("Calculating MSE...")
mse = recmetrics.mse(result2.actual, result2.predictions)
print(mse)
print("Calculated MSE!")

print("Calculating RMSE...")
rmse = recmetrics.rmse(result2.actual, result2.predictions)
print(rmse)
print("Calculated RMSE!")

# Create model (matrix of predicted values)
print("Creating model...")
algoModel = result2.pivot_table(index='userId', columns='movieId', values='predictions').fillna(0)
print("Created model!")

result2 = result2.copy().groupby('userId', as_index=False)['movieId'].agg({'actual': (lambda x: list(set(x)))})
result2 = result2.set_index("userId")

# Make recommendations for all members in the test data
print("Making recommendations for each member...")
recs = [] = []
for user in result2.index:
    predictions = get_users_predictions(user, 10, algoModel)
    recs.append(predictions)
print("Recommendations made for each member!")

result2['predictions'] = recs

nov = ratings.movieId.value_counts()
pop = dict(nov)

# Calculate novelty and personalization
print("Calculating Novelty...")
novelty, mselfinfo_list = recmetrics.novelty(recs, pop, len(users), 10)
print("Calculated Novelty!")

print("Calculating Personalization...")
personalization = recmetrics.personalization(recs)
print("Calculated Personalization!")

print("Calculating Intra-List Similarity...")
intraSim = recmetrics.intra_list_similarity(recs, movies)
print("Calculated Intra-List Similarity!")

print("Calculating Prediction Coverage...")
predCov = recmetrics.prediction_coverage(recs, catalog)
print("Calculated Prediction Coverage!")

print("Calculating Catalog Coverage...")
catCov = recmetrics.catalog_coverage(recs, catalog, 100)
print("Calculated Catalog Coverage!")

mseList.append(mse)
rmseList.append(rmse)
noveltyList.append(novelty)
personList.append(personalization)
intraSimList.append(intraSim)
predictionCoverageList.append(predCov)
catalogCoverageList.append(catCov)

# <------------------------------------------------------------------------------>

# <-------------------- SVD-KFN Hybrid Algorithm ------------------------------->

test5 = kfnAlgo.test(testSet)
test5 = pd.DataFrame(test5)
test5.drop("details", inplace=True, axis=1)
test5.columns = ['userId', 'movieId', 'actual', 'KFN predictions']
test5.drop(['userId', 'movieId', 'actual'], inplace=True, axis=1)
test5["KFN Standardized"] = ((test5['KFN predictions'] / 5) - 0.5)
print(test5.head())

result3 = pd.concat([test1, test5], axis=1, join='inner')
col = result3.loc[:, "SVD predictions": "KFN Standardized"]
result3['predictions'] = result3["SVD predictions"] + result3["KFN Standardized"]
print(result3.head())
result3.drop(["SVD predictions", "KFN Standardized"], inplace=True, axis=1)

print("Calculating MSE...")
mse = recmetrics.mse(result3.actual, result3.predictions)
print(mse)
print("Calculated MSE!")

print("Calculating RMSE...")
rmse = recmetrics.rmse(result3.actual, result3.predictions)
print(rmse)
print("Calculated RMSE!")

# Create model (matrix of predicted values)
print("Creating model...")
kfnSvdModel = result3.pivot_table(index='userId', columns='movieId', values='predictions').fillna(0)
print("Created model!")

result3 = result3.copy().groupby('userId', as_index=False)['movieId'].agg({'actual': (lambda x: list(set(x)))})
result3 = result3.set_index("userId")

# Make recommendations for all members in the test data
print("Making recommendations for each member...")
recs = [] = []
for user in result3.index:
    predictions = get_users_predictions(user, 10, kfnSvdModel)
    recs.append(predictions)
print("Recommendations made for each member!")

result3['predictions'] = recs

nov = ratings.movieId.value_counts()
pop = dict(nov)

# Calculate novelty and personalization
print("Calculating Novelty...")
novelty, mselfinfo_list = recmetrics.novelty(recs, pop, len(users), 10)
print("Calculated Novelty!")

print("Calculating Personalization...")
personalization = recmetrics.personalization(recs)
print("Calculated Personalization!")

print("Calculating Intra-List Similarity...")
intraSim = recmetrics.intra_list_similarity(recs, movies)
print("Calculated Intra-List Similarity!")

print("Calculating Prediction Coverage...")
predCov = recmetrics.prediction_coverage(recs, catalog)
print("Calculated Intra-List Similarity!")

print("Calculating Coverage...")
catCov = recmetrics.catalog_coverage(recs, catalog, 100)
print("Calculated Intra-List Similarity!")

mseList.append(mse)
rmseList.append(rmse)
noveltyList.append(novelty)
personList.append(personalization)
intraSimList.append(intraSim)
predictionCoverageList.append(predCov)
catalogCoverageList.append(catCov)

# <------------------------------------------------------------------------------>

# <-------------------- KNN-Normal Hybrid Algorithm ------------------------------->

result4 = pd.concat([test2, test4], axis=1, join='inner')
col = result4.loc[:, "KNN predictions": "Standardized"]
result4['predictions'] = result4["KNN predictions"] + result4["Standardized"]
print(result4.head())
result4.drop(["KNN predictions", "Standardized"], inplace=True, axis=1)

print("Calculating MSE...")
mse = recmetrics.mse(result4.actual, result4.predictions)
print(mse)
print("Calculated MSE!")

print("Calculating RMSE...")
rmse = recmetrics.rmse(result4.actual, result4.predictions)
print(rmse)
print("Calculated RMSE!")

# Create model (matrix of predicted values)
print("Creating model...")
algoModel = result4.pivot_table(index='userId', columns='movieId', values='predictions').fillna(0)
print("Created model!")

result4 = result4.copy().groupby('userId', as_index=False)['movieId'].agg({'actual': (lambda x: list(set(x)))})
result4 = result4.set_index("userId")

# Make recommendations for all members in the test data
print("Making recommendations for each member...")
recs = [] = []
for user in result4.index:
    predictions = get_users_predictions(user, 10, algoModel)
    recs.append(predictions)
print("Recommendations made for each member!")

result4['predictions'] = recs

nov = ratings.movieId.value_counts()
pop = dict(nov)

# Calculate novelty and personalization
print("Calculating Novelty...")
novelty, mselfinfo_list = recmetrics.novelty(recs, pop, len(users), 10)
print("Calculated Novelty!")

print("Calculating Personalization...")
personalization = recmetrics.personalization(recs)
print("Calculated Personalization!")

print("Calculating Intra-List Similarity...")
intraSim = recmetrics.intra_list_similarity(recs, movies)
print("Calculated Intra-List Similarity!")

print("Calculating Prediction Coverage...")
predCov = recmetrics.prediction_coverage(recs, catalog)
print("Calculated Intra-List Similarity!")

print("Calculating Coverage...")
catCov = recmetrics.catalog_coverage(recs, catalog, 100)
print("Calculated Intra-List Similarity!")

mseList.append(mse)
rmseList.append(rmse)
noveltyList.append(novelty)
personList.append(personalization)
intraSimList.append(intraSim)
predictionCoverageList.append(predCov)
catalogCoverageList.append(catCov)

# <------------------------------------------------------------------------------>

data = {
    'Algorithm': ["SVD", "KNN", "KNN with Means", "KNN with Z-Score", "KNN Baseline", "SVD Plus Plus",
                  "NMF", "Co-clustering", "Normal Predictor", "Baseline", "KFN", "SVD-KNN Hybrid",
                  "SVD-Normal Hybrid", "SVD-KFN Hybrid", "KNN-Normal Hybrid"],
    # Lower is better
    'MSE': mseList,
    # Lower is better
    'RMSE': rmseList,
    # Higher is better
    'Novelty': noveltyList,
    # Higher is better
    'Personalization': personList,
    # Lower is better
    'Intra-list Similarity': intraSimList,
    # Higher is better
    'Prediction Coverage': predictionCoverageList,
    # Higher is better
    'Catalog Coverage': catalogCoverageList
}

df = pd.DataFrame(data)
df.set_index("Algorithm", inplace=True, drop=True)

print("Dataset: ml-20m\n")
print(df)

df.to_csv('../Output/ML-20M/stats-ml20_sample3.csv')
