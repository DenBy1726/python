import pandas as pd
import numpy as np

r_cols = ['user_id', 'movie_id', 'rating']
m_cols = ['movie_id', 'title']
film = 'Star Wars (1977)'

ratings = pd.read_csv('./data mining/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")
movies = pd.read_csv('./data mining/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)
print(ratings.head())

movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
print(movieRatings.head())

starWarsRatings = movieRatings[film]
print(starWarsRatings.head())

similarMovies = movieRatings.corrwith(starWarsRatings)
similarMovies = similarMovies.dropna()
df = pd.DataFrame(similarMovies)
print(df.head(10))

movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
print(movieStats.head())

popularMovies = movieStats['rating']['size'] >= 100
#movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15]
df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))
df = df.sort_values(['similarity'], ascending=False)[:10]
print(df.head(10))