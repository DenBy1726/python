import pandas as pd
import numpy as np
from scipy import spatial
import operator
import importlib
import matplotlib.pyplot as plt

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('D:\\Work\\Python\\DataScience-Python3\\ml-100k\\u.data', sep='\t', names=r_cols, usecols=range(3))
ratings.head()

movieProperties = ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]})
movieProperties.head()

movieNumRatings = pd.DataFrame(movieProperties['rating']['size'])
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
movieNormalizedNumRatings.head()

movieDict = {}
with open(r'D:\\Work\\Python\\DataScience-Python3\\ml-100k\\u.item') as f:
    temp = ''
    for line in f:
        #line.decode("ISO-8859-1")
        fields = line.rstrip('\n').split('|')
        movieID = int(fields[0])
        name = fields[1]
        genres = fields[5:25]
        genres = map(int, genres)
        movieDict[movieID] = (name, np.array(list(genres)), movieNormalizedNumRatings.loc[movieID].get('size'), movieProperties.loc[movieID].rating.get('mean'))

print(movieDict[1])

def ComputeDistance(a, b, method = None):
    if(method == None):
        method = spatial.distance.cosine
    genresA = a[1]
    genresB = b[1]
    genreDistance = method(genresA, genresB)
    popularityA = a[2]
    popularityB = b[2]
    popularityDistance = abs(popularityA - popularityB)
    return genreDistance + popularityDistance
    
ComputeDistance(movieDict[2], movieDict[4])

def computeAvgRating(K,method):
    avgRating = 0
    neighbors = getNeighbors(1, K,method)
    for neighbor in neighbors:
        avgRating += movieDict[neighbor][3]
        #print (movieDict[neighbor][0] + " " + str(movieDict[neighbor][3]))
    return avgRating / K

def getNeighbors(movieID, K,method = None):
    distances = []
    for movie in movieDict:
        if (movie != movieID):
            dist = ComputeDistance(movieDict[movieID], movieDict[movie],method)
            distances.append((movie, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors


#    braycurtis       -- the Bray-Curtis distance.  
#    canberra         -- the Canberra distance.  
#    chebyshev        -- the Chebyshev distance.  
#    cityblock        -- the Manhattan distance.  
#    correlation      -- the Correlation distance.  
#    cosine           -- the Cosine distance.  
#    euclidean        -- the Euclidean distance.  
#    mahalanobis      -- the Mahalanobis distance.  
#    minkowski        -- the Minkowski distance.  
#    seuclidean       -- the normalized Euclidean distance.  
#    sqeuclidean      -- the squared Euclidean distance.  
#    wminkowski       -- (deprecated) alias of `minkowski`.

# dice(u, v[, w])	Compute the Dice dissimilarity between two boolean 1-D arrays.
# hamming(u, v[, w])	Compute the Hamming distance between two 1-D arrays.
# jaccard(u, v[, w])	Compute the Jaccard-Needham dissimilarity between two boolean 1-D arrays.
# kulsinski(u, v[, w])	Compute the Kulsinski dissimilarity between two boolean 1-D arrays.
# rogerstanimoto(u, v[, w])	Compute the Rogers-Tanimoto dissimilarity between two boolean 1-D arrays.
# russellrao(u, v[, w])	Compute the Russell-Rao dissimilarity between two boolean 1-D arrays.
# sokalmichener(u, v[, w])	Compute the Sokal-Michener dissimilarity between two boolean 1-D arrays.
# sokalsneath(u, v[, w])	Compute the Sokal-Sneath dissimilarity between two boolean 1-D arrays.
# yule(u, v[, w])	Compute the Yule dissimilarity between two boolean 1-D arrays.

actual = movieDict[1][3]
methods = [
    'braycurtis','canberra','chebyshev', 'cityblock', 'correlation', 'cosine', 'euclidean', 'minkowski', 
    'dice', 'hamming', 'jaccard', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalmichener', 'sokalsneath', 'yule'
    ]
for method in methods:
    call = getattr(spatial.distance, method)
    x = range(1,13)
    y = []
    for K in x:
        expect = computeAvgRating(K,call)
        y.append(abs(expect - actual))

    plt.plot(x,y)
    plt.savefig("./data mining/plot/knnDistance/" + str(method) + ".png", method="png")
    plt.close()


# print(computeAvgRating(spatial.distance.cosine))
# print(movieDict[1][3])
