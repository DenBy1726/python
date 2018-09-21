import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

r_cols = ['user_id', 'movie_id', 'rating']
m_cols = ['movie_id', 'title']
userId = 0

ratings = pd.read_csv('./data mining/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")
movies = pd.read_csv('./data mining/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)
print(ratings.head())

userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
print(userRatings.head())

corrMatrix = userRatings.corr(method='pearson', min_periods=100)
print(corrMatrix.head())

myRatings = userRatings.loc[0].dropna()
print(myRatings)

simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print ("Adding sims for " + myRatings.index[i] + "...")
    # Retrieve similar movies to this one that I rated
    sims = corrMatrix[myRatings.index[i]].dropna()
    # Now scale its similarity by how well I rated this movie
    sims = sims.map(lambda x: x * myRatings[i])
    # Add the score to the list of similarity candidates
    simCandidates = simCandidates.append(sims)
    
#Glance at our results so far:
print ("sorting...")
simCandidates = simCandidates.groupby(simCandidates.index).sum()
simCandidates.sort_values(inplace = True, ascending = False)
filteredSims = simCandidates.drop(myRatings.index)
print (filteredSims.head(10))