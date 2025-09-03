import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load MovieLens 100k dataset

url_ratings = "http://files.grouplens.org/datasets/movielens/ml-100k/u.data"
url_movies = "http://files.grouplens.org/datasets/movielens/ml-100k/u.item"

# Ratings: user_id, movie_id, rating, timestamp
ratings = pd.read_csv(url_ratings, sep='\t', names=['user_id','movie_id','rating','timestamp'])
# Movies: movie_id, title
movies = pd.read_csv(url_movies, sep='|', encoding='latin-1', header=None, names=['movie_id','title'] + list(range(22)))


# Create user-item rating matrix

user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')

# Fill missing ratings with user mean (simple CF approach)
user_means = user_item_matrix.mean(axis=1)
ratings_filled = user_item_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)


# Function to recommend top N movies

def recommend_movies(user_id, top_n=5):
    if user_id not in ratings_filled.index:
        print(f"User {user_id} not found!")
        return []
    
    # Get user ratings
    user_ratings = ratings_filled.loc[user_id]
    
    # Movies already rated
    rated_movies = user_item_matrix.loc[user_id].dropna().index
    
    # Predict ratings = use mean of other users for simplicity
    predicted_ratings = {}
    for movie_id in ratings_filled.columns:
        if movie_id not in rated_movies:
            predicted_ratings[movie_id] = ratings_filled[movie_id].mean()
    
    # Sort top N
    top_movies_ids = sorted(predicted_ratings, key=predicted_ratings.get, reverse=True)[:top_n]
    top_titles = [movies.set_index('movie_id').loc[movie_id, 'title'] for movie_id in top_movies_ids]
    top_scores = [predicted_ratings[movie_id] for movie_id in top_movies_ids]
    
    return top_titles, top_scores

# Example usage

user_id = int(input("Enter User ID (1-943): "))
titles, scores = recommend_movies(user_id, top_n=5)

print(f"\nTop 5 movie recommendations for user {user_id}:")
for i, (title, score) in enumerate(zip(titles, scores), 1):
    print(f"{i}. {title} (Predicted rating: {score:.2f})")

# Plot predicted ratings

plt.figure(figsize=(6,4))
plt.bar(titles, scores, color='orange')
plt.title(f"Top 5 Predicted Ratings for User {user_id}")
plt.ylabel("Predicted Rating")
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
