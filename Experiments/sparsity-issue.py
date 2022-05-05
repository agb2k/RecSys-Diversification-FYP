# Used to test Sparsity hypothesis
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

# Display maximum columns
pd.set_option("display.max_columns", None)

# Read CSV
ratings = pd.read_csv('../Datasets/BX-CSV-Dump/BX-Book-Ratings.csv', delimiter=";", encoding='latin-1',
                      names=["ID", "ISBN", "Rating"])

# Additional data cleaning
ratings["ISBN"] = ratings["ISBN"].apply(lambda x: x.strip().strip("\'").strip("\\").strip('\"').strip("\#").strip("("))
ratings.reset_index(drop=True, inplace=True)

# Find users who have rated more than 10 books
users = ratings["ID"].value_counts()
users = users[users > 10].index.tolist()

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
print("Training model...")
svdAlgo = SVD()
svdAlgo.fit(trainSet)
pickle.dump(svdAlgo, open('../Models/Book-Crossing Models/svd_algo.sav', 'wb'))
print("Model trained!")
algoList.append(svdAlgo)
algoNameList.append("SVD")

# Recommender System Algorithm, KNN
print("Training model...")
knnAlgo = KNNBasic()
knnAlgo.fit(trainSet)
pickle.dump(knnAlgo, open('../Models/Book-Crossing Models/knn_algo.sav', 'wb'))
print("Model trained!")
algoList.append(knnAlgo)
algoNameList.append("KNN")

# Recommender System Algorithm, K-Furthest Neighbours(V2)
print("Training model...")
kfnAlgo = KFN2()
kfnAlgo.fit(trainSet)
pickle.dump(kfnAlgo, open('../Models/Book-Crossing Models/KFN.sav', 'wb'))
print("Model trained!")
algoList.append(kfnAlgo)
algoNameList.append("KFN")

# Recommender System Algorithm, K-Nearest Neighbours of Furthest Neighbour
print("Training model...")
knnofnAlgo = KNNOFN()
knnofnAlgo.fit(trainSet)
pickle.dump(knnofnAlgo, open('../Models/Book-Crossing Models/KNNOFN.sav', 'wb'))
print("Model trained!")
algoList.append(knnofnAlgo)
algoNameList.append("KNNOFN")

print("Training model...")
knnofnfnAlgo = KNNOFNFN()
knnofnfnAlgo.fit(trainSet)
pickle.dump(knnofnfnAlgo, open('../Models/Book-Crossing Models/KNNOFNFN.sav', 'wb'))
print("Model trained!")
algoList.append(knnofnfnAlgo)
algoNameList.append("KNNOFNFN")

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

df.to_csv('../Output/Algorithm-Experiments-Output/Comparison-Sparsity-Issue.csv')
