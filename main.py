import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances, mean_absolute_error

# READING FILES

# Reading Ratings File
r_cols = ['user_id', 'movie_id', 'rating']
ratingsDF = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

# Reading Items File
m_cols = ['movie_id', 'title']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

# Merged Ratings and Items Files
rated_movies = pd.merge(ratingsDF, movies, on='movie_id')

# Matrix of Users and Movies
user_ratings_matrix = rated_movies.pivot_table(values='rating', index='user_id', columns='title').fillna(value=0)

# USER-BASED

# Get pearson similarities for ratings matrix
pearson_sim = 1-pairwise_distances(user_ratings_matrix, Y=None, metric="correlation", force_all_finite=True)
pd.DataFrame(pearson_sim)


# This function finds n similar users given the user_id
def find_similar_users(user_id, n):
    model_knn = NearestNeighbors(metric='correlation')
    model_knn.fit(user_ratings_matrix)

    distances, indices = model_knn.kneighbors(user_ratings_matrix.iloc[user_id-1, :].values.reshape(1, -1), n_neighbors=n+1)
    similarities = 1 - distances.flatten()

    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] + 1 == user_id:
            continue

    return similarities, indices


# This function predicts rating for specified user-item combination based on user-based approach
def predict_user_based(user_id, movie_id, n):
    prediction = 0
    similarities, indices = find_similar_users(user_id, n)
    mean_rating = user_ratings_matrix.loc[user_id, :].mean()
    sum_wt = np.sum(similarities) - 1
    product = 1
    wtd_sum = 0

    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] + 1 == user_id:
            continue
        else:
            ratings_diff = user_ratings_matrix.iloc[indices.flatten()[i], movie_id - 1] - np.mean(
                user_ratings_matrix.iloc[indices.flatten()[i], :])
            product = ratings_diff * (similarities[i])
            wtd_sum = wtd_sum + product

    prediction = int(round(mean_rating + (wtd_sum / sum_wt)))
    print('\nPredicted rating with User-Based Approach for user {0} -> item {1}: {2}'.format(user_id, movie_id, prediction))

    return prediction


# ITEM-BASED

# This function finds n similar items given the item_id
def find_similar_items(movie_id, n):
    ratings = user_ratings_matrix.T
    model_knn = NearestNeighbors(metric='cosine')
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(ratings.iloc[movie_id-1, :].values.reshape(1, -1), n_neighbors=n+1)
    similarities = 1-distances.flatten()

    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == movie_id:
            continue

    return similarities, indices


# This function predicts the rating for specified user-item combination based on item-based approach
def predict_item_based(user_id, movie_id, n):
    prediction = wtd_sum = 0
    similarities, indices = find_similar_items(movie_id, n)
    sum_wt = np.sum(similarities) - 1
    product = 1

    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] + 1 == movie_id:
            continue
        else:
            product = user_ratings_matrix.iloc[user_id - 1, indices.flatten()[i]] * (similarities[i])
            wtd_sum = wtd_sum + product
    prediction = int(round(wtd_sum / sum_wt))
    print('\nPredicted rating with Item-Based Approach for user {0} -> item {1}: {2}'.format(user_id, movie_id, prediction))

    return prediction


# Evaluating Model to Make Prediction on Ratings
def evaluate(ratings, n):
    print("Evaluation Starting...")
    n_users = ratings.shape[0]
    n_items = ratings.shape[1]
    predictionA = np.zeros((n_users, n_items))
    predictionA = pd.DataFrame(predictionA)
    predictionB = np.zeros((n_users, n_items))
    predictionB = pd.DataFrame(predictionB)
    # User-based CF (correlation)
    print("Evaluating User-based CF...")
    for i in range(n_users):
        for j in range(n_items):
            predictionA[i][j] = predict_user_based(i + 1, j + 1, n)
    mea1 = mean_absolute_error(predictionA, ratings)
    # print('Mean Absolute Error of User-based CF Approach: %.3f' % mean(mea1))

    # Item-based CF (cosine)
    print("Evaluating Item-based CF...")
    for i in range(n_users):
        for j in range(n_items):
            predictionB[i][j] = predict_item_based(i + 1, j + 1, n)
    mea2 = mean_absolute_error(predictionB, ratings)
    # print('Mean Absolute Error of Item-based CF Approach: %.3f' % mean(mea2))

    return mea1, mea2


# Finding Optimal K Value (KNN Algorithm) for User-Based Approach
def optimal_k_user():
    k_range = list(range(10, 81, 10))
    k_scores = []
    for k in k_range:
        scores = evaluate(user_ratings_matrix, k)
        k_scores.append(scores[0])
    print(k_scores)
    plt.plot(k_range, k_scores)
    plt.title('K Values For User-Based')
    plt.xlabel('Value of K for KNN')
    plt.ylabel('MAE')
    plt.show()


# Finding Optimal K Value (KNN Algorithm) for Item-Based Approach
def optimal_k_item():
    k_range = list(range(10, 81, 10))
    k_scores = []
    for k in k_range:
        scores = evaluate(user_ratings_matrix, k)
        k_scores.append(scores[1])
    print(k_scores)
    plt.plot(k_range, k_scores)
    plt.title('K Values For Item-Based')
    plt.xlabel('Value of K for KNN')
    plt.ylabel('MAE')
    plt.show()


if __name__ == '__main__':
    predict_user_based(1, 200, 80)
    predict_item_based(1, 200, 30)
