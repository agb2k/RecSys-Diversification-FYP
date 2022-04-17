import matplotlib.pyplot as plt
import pandas as pd
from surprise import SVD, SVDpp, KNNBasic, KNNWithMeans, KNNWithZScore, NMF, SlopeOne, Dataset, Reader, \
    NormalPredictor, BaselineOnly, CoClustering, KNNBaseline
from surprise.model_selection import train_test_split, GridSearchCV
import recmetrics
import pickle

trainBool = False

# Display maximum columns
pd.set_option("display.max_columns", None)

# Read CSV
ratings = pd.read_csv('../Datasets/BX-CSV-Dump/BX-Book-Ratings.csv', delimiter=";", encoding='latin-1',
                      names=["ID", "ISBN", "Rating"])

# Additional data cleaning
ratings["ISBN"] = ratings["ISBN"].apply(lambda x: x.strip().strip("\'").strip("\\").strip('\"').strip("\#").strip("("))
# ratings["Rating"] = ratings['Rating'].str.replace("\"", "").astype(int)
ratings.reset_index(drop=True, inplace=True)

# Find users who have rated more than 1000 books
users = ratings["ID"].value_counts()
users = users[users > 1000].index.tolist()

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
trainSet, testSet = train_test_split(data, test_size=0.2)
print("Train-test split complete!")

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

# Recommender System Algorithm, Slope One
if not trainBool:
    print("Loading model...")
    slopeAlgo = pickle.load(open('../Models/Book-Crossing Models/slope_algo.sav', 'rb'))
    print("Model loaded!")
else:
    print("Training model...")
    slopeAlgo = SlopeOne()
    slopeAlgo.fit(trainSet)
    pickle.dump(slopeAlgo, open('../Models/Book-Crossing Models/slope_algo.sav', 'wb'))
    print("Model trained!")

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

algoList = [svdAlgo, knnAlgo, knnMeansAlgo, knnZAlgo, knnBaselineAlgo, svdPPAlgo, nmfAlgo, slopeAlgo, ccAlgo,
            baselineAlgo, normalAlgo]

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

# <----------------------------- SVD-KNN Hybrid Algorithm ---------------------------------------------->
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

# <------------------------------------------------------------------------------>

data = {
    'Algorithm': ["SVD", "KNN", "KNN with Means", "KNN with Z-Score", "KNN Baseline", "SVD Plus Plus",
                  "NMF", "Slope One", "Co-Clustering", "Baseline", "Normal Predictor",
                  "SVD-KNN Hybrid"],
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

df = pd.DataFrame(data)
df.set_index("Algorithm", inplace=True, drop=True)

print("Dataset: Book-Crossing\n")
print(df)

df.to_csv('Output/Book-Crossing/stats-bc.csv')
