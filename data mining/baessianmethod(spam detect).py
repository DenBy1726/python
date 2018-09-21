import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

data = DataFrame({'message': [], 'class': []})
test = DataFrame({'message': [], 'class': []})
learn = DataFrame({'message': [], 'class': []})

spam = dataFrameFromDirectory('./data mining/emails/spam', 'spam')
ham = dataFrameFromDirectory('./data mining/emails/ham', 'ham')

data = data.append(spam)
data = data.append(ham)

print(data.head())

# vectorizer = CountVectorizer()
# counts = vectorizer.fit_transform(data['message'].values)

# classifier = MultinomialNB()
# targets = data['class'].values
# classifier.fit(counts, targets)

# examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]
# example_counts = vectorizer.transform(examples)
# predictions = classifier.predict(example_counts)
# print(predictions)

learn = learn.append(spam[:int(len(spam)*0.8)])
learn = learn.append(ham[:int(len(ham)*0.8)])

test = test.append(spam[int(len(spam)*0.8):])
test = test.append(ham[int(len(ham)*0.8):])

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(learn['message'].values)
classifier = MultinomialNB()
targets = learn['class'].values
classifier.fit(counts, targets)

example_counts = vectorizer.transform(test['message'].values)
predictions = classifier.predict(example_counts)

match = 0
for i in range(len(predictions)):
    if(predictions[i] == test['class'].values[i]):
        match+=1

print("percentage is " + int(str(match/len(predictions)*100)) + "%")

examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)
