from pylab import dot

def mean(values):
    if(len(values) == 0):
        return None
    return sum(values)/len(values)

def median(values):
   n = len(values)
   if n < 1:
        return None
   if n % 2 == 1:
       return sorted(values)[n//2]
   else:
       return sum(sorted(values)[n//2-1:n//2+1])/2.0

def mode(values):
    if(len(values) == 0):
        return None
    return max(set(values), key=values.count)

def variance(values, population = True):
    if(len(values) < 2):
        return None
    _mean = mean(values)
    diffs = [x - _mean for x in values]   
    diffs = [x*x for x in diffs]
    if(population):
        return mean(diffs)
    else:
        return sum(diffs)/(len(values) - 1)

def deviation(value,population = True):
    if (isinstance(value, list)):
        value = variance(value,population) 
    return value**(1/2)
    
def de_mean(x):
    xmean = mean(x)
    return [xi - xmean for xi in x]

def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n-1)

def correlation(x, y):
    stddevx = deviation(x)
    stddevy = deviation(y)
    return covariance(x,y) / stddevx / stddevy


print(mean([1,2,3,4,5]))
print(mean([1,2]))
print(mean([0,2,6,2,0,1,1,0]))

print()

print(median([1,2,3,4,5]))
print(median([1,2]))
print(median([0,2,6,2,0,1,1,0]))

print()

print(mode([1,2,3,4,5]))
print(mode([1,2]))
print(mode([0,2,6,2,0,1,1,0]))

print()

print(variance([1,2,3,4,5]))
print(variance([1,2]))
print(variance([0,2,6,2,0,1,1,0]))
print(variance([1,4,5,4,8]))

print()

print(variance([1,2,3,4,5],population=False))
print(variance([1,2],population=False))
print(variance([0,2,6,2,0,1,1,0],population=False))
print(variance([1,4,5,4,8],population=False))

print()

print(deviation([1,2,3,4,5]))
print(deviation([1,2]))
print(deviation([0,2,6,2,0,1,1,0]))
print(deviation([1,4,5,4,8]))

print()

print(deviation([1,2,3,4,5],population=False))
print(deviation([1,2],population=False))
print(deviation([0,2,6,2,0,1,1,0],population=False))
print(deviation([1,4,5,4,8],population=False))