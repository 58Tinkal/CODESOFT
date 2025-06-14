import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load datasets from local files
ratings = pd.read_csv("ratings.csv")
print(ratings.head())

movies = pd.read_csv("movies.csv")
print(movies.head())

# Dataset statistics
n_ratings = len(ratings)
n_movies = len(ratings['movieId'].unique())
n_users = len(ratings['userId'].unique())

print(f"Number of ratings: {n_ratings}")
print(f"Number of unique movieId's: {n_movies}")
print(f"Number of unique users: {n_users}")
print(f"Average ratings per user: {round(n_ratings/n_users, 2)}")
print(f"Average ratings per movie: {round(n_ratings/n_movies, 2)}")

# User rating frequency
user_freq = ratings[['userId', 'movieId']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'n_ratings']
print(user_freq.head())

# Movie stats
mean_rating = ratings.groupby('movieId')[['rating']].mean()
lowest_rated = mean_rating['rating'].idxmin()
highest_rated = mean_rating['rating'].idxmax()

print("\nLowest Rated Movie:")
print(movies.loc[movies['movieId'] == lowest_rated])

print("\nHighest Rated Movie:")
print(movies.loc[movies['movieId'] == highest_rated])

movie_stats = ratings.groupby('movieId')[['rating']].agg(['count', 'mean'])
movie_stats.columns = movie_stats.columns.droplevel()

# Create sparse matrix
def create_matrix(df):
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))

    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

# Find similar movies
def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    neighbour_ids = []

    if movie_id not in movie_mapper:
        print(f"Movie ID {movie_id} not found.")
        return []

    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    movie_vec = movie_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)

    for i in range(0, k):
        n = neighbour[1].item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids

# Map movie IDs to titles
movie_titles = dict(zip(movies['movieId'], movies['title']))

# Recommend similar movies based on one movie
movie_id = 3
similar_ids = find_similar_movies(movie_id, X, k=10)
movie_title = movie_titles.get(movie_id, "Movie not found")

print(f"\nSince you watched {movie_title}")
for i in similar_ids:
    print(movie_titles.get(i, "Movie not found"))

# Recommend movies for a user
def recommend_movies_for_user(user_id, X, user_mapper, movie_mapper, movie_inv_mapper, k=10):
    df1 = ratings[ratings['userId'] == user_id]

    if df1.empty:
        print(f"\nUser with ID {user_id} does not exist.")
        return

    movie_id = df1[df1['rating'] == max(df1['rating'])]['movieId'].iloc[0]
    movie_title = movie_titles.get(movie_id, "Movie not found")
    similar_ids = find_similar_movies(movie_id, X, k)

    print(f"\nSince you watched {movie_title}, you might also like:")
    for i in similar_ids:
        print(movie_titles.get(i, "Movie not found"))

# Recommend for specific users
recommend_movies_for_user(150, X, user_mapper, movie_mapper, movie_inv_mapper, k=10)
recommend_movies_for_user(2300, X, user_mapper, movie_mapper, movie_inv_mapper, k=10)
