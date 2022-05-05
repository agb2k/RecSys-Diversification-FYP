# Main experiments folder for book-crossing dataset
import matplotlib.pyplot as plt
import pandas as pd
from surprise import SVD, SVDpp, KNNBasic, KNNWithMeans, KNNWithZScore, NMF, Dataset, Reader, \
    NormalPredictor, BaselineOnly, CoClustering, KNNBaseline
from surprise.model_selection import train_test_split, GridSearchCV
import recmetrics
import pickle

from Algorithms.KFN2 import KFN2
from Algorithms.KNNOFN import KNNOFN
from Algorithms.KNNOFNFN import KNNOFNFN
from Algorithms.KNNONN import KNNONN

trainBool = False
trainNew = False

# Display maximum columns
pd.set_option("display.max_columns", None)

# Read CSV
ratings = pd.read_csv('../Datasets/BX-CSV-Dump/BX-Book-Ratings.csv', delimiter=";", encoding='latin-1',
                      names=["ID", "ISBN", "Rating"])

# Additional data cleaning
ratings["ISBN"] = ratings["ISBN"].apply(lambda x: x.strip().strip("\'").strip("\\").strip('\"').strip("\#").strip("("))
ratings.reset_index(drop=True, inplace=True)

# Find users who have rated more than 100 books
users = ratings["ID"].value_counts()
users = users[users > 100].index.tolist()

# Filter ratings according to corresponding users
ratings = ratings.query('ID in @users')
rated_books = ratings["ISBN"].tolist()

# Find corresponding rated books
books = pd.read_csv('../Datasets/BX-CSV-Dump/BX-Books.csv', delimiter=";", escapechar='\\', encoding='latin-1',
                    names=['ISBN', 'Title', 'Author', 'Year', 'Publisher', 'URL1', 'URL2', 'URL3'], skiprows=1)

books = books.query('ISBN in @rated_books')
books.set_index("ISBN", inplace=True, drop=True)

# Preprocessing movie genre
books.drop(["Author", "Publisher", "URL1", "URL2", "URL3"], axis=1, inplace=True)
books.reset_index(inplace=True)

# Remove duplicate rows
books.drop_duplicates("ISBN", inplace=True)
books.set_index('ISBN', inplace=True)

# Unique book ISBN from ratings
bookCatalog = books.index.unique().tolist()

# Checks if ISBN from ratings are valid books
ratings = ratings[ratings['ISBN'].isin(bookCatalog)]

# Removes ratings of 0
ratings.drop(ratings.index[ratings['Rating'] == 0], inplace=True)
ratings.drop(ratings.index[ratings['Rating'] == '0'], inplace=True)

# Reshaping into dummy code
books = pd.get_dummies(books.Year, prefix='Year')

# Getting unique ISBNs
catalog = ratings.ISBN.unique().tolist()

# Long Tail Plot for existing dataset ratings to show popularity of books
fig = plt.figure(figsize=(15, 7))
recmetrics.long_tail_plot(df=ratings, item_id_column="ISBN", interaction_type="book ratings",
                          percentage=0.5, x_labels=False)

# Recommender system

# Preparing data for system
print("Performing train-test split...")
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[['ID', 'ISBN', 'Rating']], reader)
trainSet, testSet = train_test_split(data, test_size=0.2, random_state=10101)
print("Train-test split complete!")

algoList = []
algoNameList = []
# Recommender System Algorithm, SVD
if not trainBool:
    print("Loading model...")
    svdAlgo = pickle.load(open('../Models/Book-Crossing Models/svd_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    svdAlgo = SVD()
    svdAlgo.fit(trainSet)
    pickle.dump(svdAlgo, open('../Models/Book-Crossing Models/svd_algo.sav', 'wb'))
    print("Model trained!")
algoList.append(svdAlgo)
algoNameList.append("SVD")

# Recommender System Algorithm, KNN
if not trainBool:
    print("Loading model...")
    knnAlgo = pickle.load(open('../Models/Book-Crossing Models/knn_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    knnAlgo = KNNBasic()
    knnAlgo.fit(trainSet)
    pickle.dump(knnAlgo, open('../Models/Book-Crossing Models/knn_algo.sav', 'wb'))
    print("Model trained!")
algoList.append(knnAlgo)
algoNameList.append("KNN")

# Recommender System Algorithm, KNN with Means
if not trainBool:
    print("Loading model...")
    knnMeansAlgo = pickle.load(open('../Models/Book-Crossing Models/knn_means_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    knnMeansAlgo = KNNWithMeans()
    knnMeansAlgo.fit(trainSet)
    pickle.dump(knnMeansAlgo, open('../Models/Book-Crossing Models/knn_means_algo.sav', 'wb'))
    print("Model trained!")
algoList.append(knnMeansAlgo)
algoNameList.append("KNN with Means")

# Recommender System Algorithm, KNN with Z-Score
if not trainBool:
    print("Loading model...")
    knnZAlgo = pickle.load(open('../Models/Book-Crossing Models/knn_Z_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    knnZAlgo = KNNWithZScore()
    knnZAlgo.fit(trainSet)
    pickle.dump(knnZAlgo, open('../Models/Book-Crossing Models/knn_Z_algo.sav', 'wb'))
    print("Model trained!")
algoList.append(knnZAlgo)
algoNameList.append("KNN with Z-Score")

# Recommender System Algorithm, KNN Baseline
if not trainBool:
    print("Loading model...")
    knnBaselineAlgo = pickle.load(open('../Models/Book-Crossing Models/knn_baseline_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    knnBaselineAlgo = KNNBaseline()
    knnBaselineAlgo.fit(trainSet)
    pickle.dump(knnBaselineAlgo, open('../Models/Book-Crossing Models/knn_baseline_algo.sav', 'wb'))
    print("Model trained!")
algoList.append(knnBaselineAlgo)
algoNameList.append("KNN Baseline")

# Recommender System Algorithm, SVD plus plus
if not trainBool:
    print("Loading model...")
    svdPPAlgo = pickle.load(open('../Models/Book-Crossing Models/svd_pp_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    svdPPAlgo = SVDpp()
    svdPPAlgo.fit(trainSet)
    pickle.dump(svdPPAlgo, open('../Models/Book-Crossing Models/svd_pp_algo.sav', 'wb'))
    print("Model trained!")
algoList.append(svdPPAlgo)
algoNameList.append("SVD Plus Plus")

# Recommender System Algorithm, Non-Negative Matrix Factorization
if not trainBool:
    print("Loading model...")
    nmfAlgo = pickle.load(open('../Models/Book-Crossing Models/nmf_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    nmfAlgo = NMF()
    nmfAlgo.fit(trainSet)
    pickle.dump(nmfAlgo, open('../Models/Book-Crossing Models/nmf_algo.sav', 'wb'))
    print("Model trained!")
algoList.append(nmfAlgo)
algoNameList.append("Non-Negative Matrix Factorization")

# Recommender System Algorithm, Co-clustering
if not trainBool:
    print("Loading model...")
    ccAlgo = pickle.load(open('../Models/Book-Crossing Models/cc_algo', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    ccAlgo = CoClustering()
    ccAlgo.fit(trainSet)
    pickle.dump(ccAlgo, open('../Models/Book-Crossing Models/cc_algo', 'wb'))
    print("Model trained!")
algoList.append(ccAlgo)
algoNameList.append("Co-clustering")

# Recommender System Algorithm, Normal Predictor (Random)
if not trainBool:
    print("Loading model...")
    normalAlgo = pickle.load(open('../Models/Book-Crossing Models/normal_algo', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    normalAlgo = NormalPredictor()
    normalAlgo.fit(trainSet)
    pickle.dump(normalAlgo, open('../Models/Book-Crossing Models/normal_algo', 'wb'))
    print("Model trained!")
algoList.append(normalAlgo)
algoNameList.append("Normal Predictor")

# Recommender System Algorithm, Baseline
if not trainBool:
    print("Loading model...")
    baselineAlgo = pickle.load(open('../Models/Book-Crossing Models/baseline_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    baselineAlgo = BaselineOnly()
    baselineAlgo.fit(trainSet)
    pickle.dump(baselineAlgo, open('../Models/Book-Crossing Models/baseline_algo.sav', 'wb'))
    print("Model trained!")
algoList.append(baselineAlgo)
algoNameList.append("Baseline")

# Recommender System Algorithm, K-Furthest Neighbours(V2)
if not trainNew:
    print("Loading model...")
    kfnAlgo = pickle.load(open('../Models/Book-Crossing Models/KFN.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    kfnAlgo = KFN2()
    kfnAlgo.fit(trainSet)
    pickle.dump(kfnAlgo, open('../Models/Book-Crossing Models/KFN.sav', 'wb'))
    print("Model trained!")
algoList.append(kfnAlgo)
algoNameList.append("KFN")

# Recommender System Algorithm, K-Nearest Neighbours of Furthest Neighbour
if not trainNew:
    print("Loading model...")
    knnofnAlgo = pickle.load(open('../Models/Book-Crossing Models/KNNOFN.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    knnofnAlgo = KNNOFN()
    knnofnAlgo.fit(trainSet)
    pickle.dump(knnofnAlgo, open('../Models/Book-Crossing Models/KNNOFN.sav', 'wb'))
    print("Model trained!")
algoList.append(knnofnAlgo)
algoNameList.append("KNNOFN")

# Recommender System Algorithm, K-Nearest Neighbours of Furthest Neighbours Furthest Neighbours
if not trainNew:
    print("Loading model...")
    knnofnfnAlgo = pickle.load(open('../Models/Book-Crossing Models/KNNOFNFN.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    knnofnfnAlgo = KNNOFNFN()
    knnofnfnAlgo.fit(trainSet)
    pickle.dump(knnofnfnAlgo, open('../Models/Book-Crossing Models/KNNOFNFN.sav', 'wb'))
    print("Model trained!")
algoList.append(knnofnfnAlgo)
algoNameList.append("KNNOFNFN")

# # Recommender System Algorithm, K-Nearest Neighbours of Nearest Neighbours
# if not trainNew:
#     print("Loading model...")
#     knnonnAlgo = pickle.load(open('../Models/Book-Crossing Models/KNNONN.sav', 'rb'))
#     print("Model loaded!")
# else:
#     print("Training model...")
#     knnonnAlgo = KNNONN()
#     knnonnAlgo.fit(trainSet)
#     pickle.dump(knnonnAlgo, open('../Models/Book-Crossing Models/KNNONN.sav', 'wb'))
#     print("Model trained!")
# algoList.append(knnonnAlgo)
# algoNameList.append("KNNONN")

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
    test.columns = ['ID', 'ISBN', 'actual', 'predictions']

    # Calculate mse and rmse
    print("Calculating MSE...")
    mse = recmetrics.mse(test.actual, test.predictions)
    print("Calculated MSE!")

    print("Calculating RMSE...")
    rmse = recmetrics.rmse(test.actual, test.predictions)
    print("Calculated RMSE!")

    # Create model (matrix of predicted values)
    print("Creating model...")
    algoModel = test.pivot_table(index='ID', columns='ISBN', values='predictions').fillna(0)
    print("Created model!")


    # Gets user predictions and returns it in a list form
    def get_users_predictions(user_id, n, model):
        recommended_items = pd.DataFrame(model.loc[user_id])
        recommended_items.columns = ["predicted_rating"]
        recommended_items = recommended_items.sort_values('predicted_rating', ascending=False)
        recommended_items = recommended_items.head(n)
        return recommended_items.index.tolist()


    test = test.copy().groupby('ID', as_index=False)['ISBN'].agg({'actual': (lambda x: list(set(x)))})
    test = test.set_index("ID")

    # Make recommendations for all members in the test data
    print("Making recommendations for each member...")
    recs = [] = []
    for user in test.index:
        predictions = get_users_predictions(user, 10, algoModel)
        recs.append(predictions)
    print("Recommendations made for each member!")

    test['predictions'] = recs

    nov = ratings.ISBN.value_counts()
    pop = dict(nov)

    # Calculate novelty and personalization
    print("Calculating Novelty...")
    novelty, mselfinfo_list = recmetrics.novelty(recs, pop, len(users), 10)
    print("Calculated Novelty!")

    print("Calculating Personalization...")
    personalization = recmetrics.personalization(recs)
    print("Calculated Personalization!")

    print("Calculating Intra-List Similarity...")
    intraSim = recmetrics.intra_list_similarity(recs, books)
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

    print(algo)

    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"Novelty: {novelty}")
    print(f"Personalization: {personalization}")
    print(f"Intra-list Similarity: {intraSim}")
    print(f"Prediction Coverage: {predCov}")
    print(f"Catalog Coverage: {catCov}")

# <----------------------------- SVD-KNN Hybrid Algorithm ---------------------------------------------->
test1 = svdAlgo.test(testSet)
test1 = pd.DataFrame(test1)
test1.drop("details", inplace=True, axis=1)
test1.columns = ['ID', 'ISBN', 'actual', 'SVD predictions']

test2 = knnAlgo.test(testSet)
test2 = pd.DataFrame(test2)
test2.drop("details", inplace=True, axis=1)
test2.columns = ['ID', 'ISBN', 'actual', 'KNN predictions']
test2.drop(['ID', 'ISBN', 'actual'], inplace=True, axis=1)

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
algoModel = result.pivot_table(index='ID', columns='ISBN', values='predictions').fillna(0)
print("Created model!")


# Gets user predictions and returns it in a list form
def get_users_predictions(user_id, n, model):
    recommended_items = pd.DataFrame(model.loc[user_id])
    recommended_items.columns = ["predicted_rating"]
    recommended_items = recommended_items.sort_values('predicted_rating', ascending=False)
    recommended_items = recommended_items.head(n)
    return recommended_items.index.tolist()


result = result.copy().groupby('ID', as_index=False)['ISBN'].agg({'actual': (lambda x: list(set(x)))})
result = result.set_index("ID")

# Make recommendations for all members in the test data
print("Making recommendations for each member...")
recs = [] = []
for user in result.index:
    predictions = get_users_predictions(user, 10, algoModel)
    recs.append(predictions)
print("Recommendations made for each member!")

result['predictions'] = recs

nov = ratings.ISBN.value_counts()
pop = dict(nov)

# Calculate novelty and personalization
print("Calculating Novelty...")
novelty, mselfinfo_list = recmetrics.novelty(recs, pop, len(users), 10)
print("Calculated Novelty!")

print("Calculating Personalization...")
personalization = recmetrics.personalization(recs)
print("Calculated Personalization!")

print("Calculating Intra-List Similarity...")
intraSim = recmetrics.intra_list_similarity(recs, books)
print("Calculated Intra-List Similarity!")

print("Calculating Prediction Coverage...")
predCov = recmetrics.prediction_coverage(recs, catalog)
print("Calculated Intra-List Similarity!")

print("Calculating Coverage...")
catCov = recmetrics.catalog_coverage(recs, catalog, 100)
print("Calculated Intra-List Similarity!")

algoNameList.append("SVD-KNN Hybrid")
mseList.append(mse)
rmseList.append(rmse)
noveltyList.append(novelty)
personList.append(personalization)
intraSimList.append(intraSim)
predictionCoverageList.append(predCov)
catalogCoverageList.append(catCov)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"Novelty: {novelty}")
print(f"Personalization: {personalization}")
print(f"Intra-list Similarity: {intraSim}")
print(f"Prediction Coverage: {predCov}")
print(f"Catalog Coverage: {catCov}")

# <------------------------------------------------------------------------------>

# <-------------------- SVD-Normal Hybrid Algorithm ------------------------------->
test4 = normalAlgo.test(testSet)
test4 = pd.DataFrame(test4)
test4.drop("details", inplace=True, axis=1)
test4.columns = ['ID', 'ISBN', 'actual', 'Normal predictions']
test4.drop(['ID', 'ISBN', 'actual'], inplace=True, axis=1)
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
algoModel = result2.pivot_table(index='ID', columns='ISBN', values='predictions').fillna(0)
print("Created model!")

result2 = result2.copy().groupby('ID', as_index=False)['ISBN'].agg({'actual': (lambda x: list(set(x)))})
result2 = result2.set_index("ID")

# Make recommendations for all members in the test data
print("Making recommendations for each member...")
recs = [] = []
for user in result2.index:
    predictions = get_users_predictions(user, 10, algoModel)
    recs.append(predictions)
print("Recommendations made for each member!")

result2['predictions'] = recs

nov = ratings.ISBN.value_counts()
pop = dict(nov)

# Calculate novelty and personalization
print("Calculating Novelty...")
novelty, mselfinfo_list = recmetrics.novelty(recs, pop, len(users), 10)
print("Calculated Novelty!")

print("Calculating Personalization...")
personalization = recmetrics.personalization(recs)
print("Calculated Personalization!")

print("Calculating Intra-List Similarity...")
intraSim = recmetrics.intra_list_similarity(recs, books)
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

algoNameList.append("SVD-Normal Hybrid")

# <------------------------------------------------------------------------------>

# <-------------------- KNN-Normal Hybrid Algorithm ------------------------------->
test5 = knnAlgo.test(testSet)
test5 = pd.DataFrame(test5)
test5.drop("details", inplace=True, axis=1)
test5.columns = ['ID', 'ISBN', 'actual', 'KNN predictions']

# test4 = normalAlgo.test(testSet)
# test4 = pd.DataFrame(test4)
# test4.drop("details", inplace=True, axis=1)
# test4.columns = ['ID', 'movieId', 'actual', 'Normal predictions']
# test4.drop(['ID', 'movieId', 'actual'], inplace=True, axis=1)
# test4["Standardized"] = ((test4['Normal predictions'] / 5) - 0.5) * -1
# print(test4.head())

result2 = pd.concat([test4, test5], axis=1, join='inner')
col = result2.loc[:, "KNN predictions": "Standardized"]
result2['predictions'] = result2["KNN predictions"] + result2["Standardized"]
print(result2.head())
result2.drop(["KNN predictions", "Standardized"], inplace=True, axis=1)

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
algoModel = result2.pivot_table(index='ID', columns='ISBN', values='predictions').fillna(0)
print("Created model!")

result2 = result2.copy().groupby('ID', as_index=False)['ISBN'].agg({'actual': (lambda x: list(set(x)))})
result2 = result2.set_index("ID")

# Make recommendations for all members in the test data
print("Making recommendations for each member...")
recs = [] = []
for user in result2.index:
    predictions = get_users_predictions(user, 10, algoModel)
    recs.append(predictions)
print("Recommendations made for each member!")

result2['predictions'] = recs

nov = ratings.ISBN.value_counts()
pop = dict(nov)

# Calculate novelty and personalization
print("Calculating Novelty...")
novelty, mselfinfo_list = recmetrics.novelty(recs, pop, len(users), 10)
print("Calculated Novelty!")

print("Calculating Personalization...")
personalization = recmetrics.personalization(recs)
print("Calculated Personalization!")

print("Calculating Intra-List Similarity...")
intraSim = recmetrics.intra_list_similarity(recs, books)
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

algoNameList.append("KNN-Inverse Normal Hybrid")

# <------------------------------------------------------------------------------>

# <-------------------- SVD-KFN Hybrid Algorithm ------------------------------->

test5 = kfnAlgo.test(testSet)
test5 = pd.DataFrame(test5)
test5.drop("details", inplace=True, axis=1)
test5.columns = ['ID', 'ISBN', 'actual', 'KFN predictions']
test5.drop(['ID', 'ISBN', 'actual'], inplace=True, axis=1)
test5["KFN Standardized"] = ((test5['KFN predictions'] / 2.5) - 1)
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
kfnSvdModel = result3.pivot_table(index='ID', columns='ISBN', values='predictions').fillna(0)
print("Created model!")

result3 = result3.copy().groupby('ID', as_index=False)['ISBN'].agg({'actual': (lambda x: list(set(x)))})
result3 = result3.set_index("ID")

# Make recommendations for all members in the test data
print("Making recommendations for each member...")
recs = [] = []
for user in result3.index:
    predictions = get_users_predictions(user, 10, kfnSvdModel)
    recs.append(predictions)
print("Recommendations made for each member!")

result3['predictions'] = recs

nov = ratings.ISBN.value_counts()
pop = dict(nov)

# Calculate novelty and personalization
print("Calculating Novelty...")
novelty, mselfinfo_list = recmetrics.novelty(recs, pop, len(users), 10)
print("Calculated Novelty!")

print("Calculating Personalization...")
personalization = recmetrics.personalization(recs)
print("Calculated Personalization!")

print("Calculating Intra-List Similarity...")
intraSim = recmetrics.intra_list_similarity(recs, books)
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

algoNameList.append("SVD-KFN Hybrid")

# <------------------------------------------------------------------------------>

# <-------------------- SVD-KNNOFN Hybrid Algorithm ------------------------------->

test7 = knnofnAlgo.test(testSet)
test7 = pd.DataFrame(test7)
test7.drop("details", inplace=True, axis=1)
test7.columns = ['ID', 'ISBN', 'actual', 'KNNOFN predictions']
test7.drop(['ID', 'ISBN', 'actual'], inplace=True, axis=1)
print(test7.head())

# mean
result5 = pd.concat([test1, test7], axis=1, join='inner')
col = result5.loc[:, "SVD predictions": "KNNOFN predictions"]
result5['predictions'] = col.mean(axis=1)
result5.drop(["SVD predictions", "KNNOFN predictions"], inplace=True, axis=1)

print("Calculating MSE...")
mse = recmetrics.mse(result5.actual, result5.predictions)
print("Calculated MSE!")

print("Calculating RMSE...")
rmse = recmetrics.rmse(result5.actual, result5.predictions)
print("Calculated RMSE!")

# Create model (matrix of predicted values)
print("Creating model...")
knnofnSvdModel = result5.pivot_table(index='ID', columns='ISBN', values='predictions').fillna(0)
print("Created model!")

result5 = result5.copy().groupby('ID', as_index=False)['ISBN'].agg({'actual': (lambda x: list(set(x)))})
result5 = result5.set_index("ID")

# Make recommendations for all members in the test data
print("Making recommendations for each member...")
recs = [] = []
for user in result5.index:
    predictions = get_users_predictions(user, 10, knnofnSvdModel)
    recs.append(predictions)
print("Recommendations made for each member!")

result5['predictions'] = recs

nov = ratings.ISBN.value_counts()
pop = dict(nov)

# Calculate novelty and personalization
print("Calculating Novelty...")
novelty, mselfinfo_list = recmetrics.novelty(recs, pop, len(users), 10)
print("Calculated Novelty!")

print("Calculating Personalization...")
personalization = recmetrics.personalization(recs)
print("Calculated Personalization!")

print("Calculating Intra-List Similarity...")
intraSim = recmetrics.intra_list_similarity(recs, books)
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

algoNameList.append("SVD-KNNOFN Hybrid")

# <------------------------------------------------------------------------------>

# <-------------------- SVD-KNNOFN Hybrid Algorithm V2------------------------------->

test7 = knnofnAlgo.test(testSet)
test7 = pd.DataFrame(test7)
test7.drop("details", inplace=True, axis=1)
test7.columns = ['ID', 'ISBN', 'actual', 'KNNOFN predictions']
test7.drop(['ID', 'ISBN', 'actual'], inplace=True, axis=1)
print(test7.head())

# mean
result5 = pd.concat([test1, test7], axis=1, join='inner')
col = result5.loc[:, "SVD predictions": "KNNOFN predictions"]
result5['predictions'] = 0.25 * result5["SVD predictions"] + 0.75 * result5["KNNOFN predictions"]
result5.drop(["SVD predictions", "KNNOFN predictions"], inplace=True, axis=1)

print("Calculating MSE...")
mse = recmetrics.mse(result5.actual, result5.predictions)
print("Calculated MSE!")

print("Calculating RMSE...")
rmse = recmetrics.rmse(result5.actual, result5.predictions)
print("Calculated RMSE!")

# Create model (matrix of predicted values)
print("Creating model...")
knnofnSvdModel = result5.pivot_table(index='ID', columns='ISBN', values='predictions').fillna(0)
print("Created model!")

result5 = result5.copy().groupby('ID', as_index=False)['ISBN'].agg({'actual': (lambda x: list(set(x)))})
result5 = result5.set_index("ID")

# Make recommendations for all members in the test data
print("Making recommendations for each member...")
recs = [] = []
for user in result5.index:
    predictions = get_users_predictions(user, 10, knnofnSvdModel)
    recs.append(predictions)
print("Recommendations made for each member!")

result5['predictions'] = recs

nov = ratings.ISBN.value_counts()
pop = dict(nov)

# Calculate novelty and personalization
print("Calculating Novelty...")
novelty, mselfinfo_list = recmetrics.novelty(recs, pop, len(users), 10)
print("Calculated Novelty!")

print("Calculating Personalization...")
personalization = recmetrics.personalization(recs)
print("Calculated Personalization!")

print("Calculating Intra-List Similarity...")
intraSim = recmetrics.intra_list_similarity(recs, books)
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

algoNameList.append("SVD-KNNOFN Hybrid V2")

# <------------------------------------------------------------------------------>

# <-------------------- SVD-KNNOFN Hybrid Algorithm V3------------------------------->

test7 = knnofnAlgo.test(testSet)
test7 = pd.DataFrame(test7)
test7.drop("details", inplace=True, axis=1)
test7.columns = ['ID', 'ISBN', 'actual', 'KNNOFN predictions']
test7.drop(['ID', 'ISBN', 'actual'], inplace=True, axis=1)
print(test7.head())

# mean
result5 = pd.concat([test1, test7], axis=1, join='inner')
col = result5.loc[:, "SVD predictions": "KNNOFN predictions"]
result5['predictions'] = 0.2 * result5["SVD predictions"] + 0.8 * result5["KNNOFN predictions"]
result5.drop(["SVD predictions", "KNNOFN predictions"], inplace=True, axis=1)

print("Calculating MSE...")
mse = recmetrics.mse(result5.actual, result5.predictions)
print("Calculated MSE!")

print("Calculating RMSE...")
rmse = recmetrics.rmse(result5.actual, result5.predictions)
print("Calculated RMSE!")

# Create model (matrix of predicted values)
print("Creating model...")
knnofnSvdModel = result5.pivot_table(index='ID', columns='ISBN', values='predictions').fillna(0)
print("Created model!")

result5 = result5.copy().groupby('ID', as_index=False)['ISBN'].agg({'actual': (lambda x: list(set(x)))})
result5 = result5.set_index("ID")

# Make recommendations for all members in the test data
print("Making recommendations for each member...")
recs = [] = []
for user in result5.index:
    predictions = get_users_predictions(user, 10, knnofnSvdModel)
    recs.append(predictions)
print("Recommendations made for each member!")

result5['predictions'] = recs

nov = ratings.ISBN.value_counts()
pop = dict(nov)

# Calculate novelty and personalization
print("Calculating Novelty...")
novelty, mselfinfo_list = recmetrics.novelty(recs, pop, len(users), 10)
print("Calculated Novelty!")

print("Calculating Personalization...")
personalization = recmetrics.personalization(recs)
print("Calculated Personalization!")

print("Calculating Intra-List Similarity...")
intraSim = recmetrics.intra_list_similarity(recs, books)
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

algoNameList.append("SVD-KNNOFN Hybrid V3")

# <------------------------------------------------------------------------------>

data = {
    'Algorithm': algoNameList,
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
    # # Higher is better
    'Prediction Coverage': predictionCoverageList,
    # Higher is better
    'Catalog Coverage': catalogCoverageList
}

print(data)

for x in data.values():
    print(f"{len(x)}\n")


df = pd.DataFrame(data)
df.set_index("Algorithm", inplace=True, drop=True)

print("Dataset: Book-Crossing\n")
print(df)

df.to_csv('../Output/Book-Crossing/stats-bc_sample9.csv')
