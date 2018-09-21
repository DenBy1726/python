import numpy as np
from numpy import array
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import tree

from IPython.display import Image  
from sklearn.externals.six import StringIO  
from pydotplus import graph_from_dot_data 
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

def create_model():
    model = Sequential()
    #4 feature inputs going into an 6-unit layer (more does not seem to help - in fact you can go down to 4)
    model.add(Dense(6, input_dim=4, kernel_initializer='normal', activation='relu'))
    # "Deep learning" turns out to be unnecessary - this additional hidden layer doesn't help either.
    #model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    # Output layer with a binary classification (benign or malignant)
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model; adam seemed to work best
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


columns = ["BI_RADS", "age", "shape", "margin", "density","severity"]

df = pd.read_csv("./data mining/mammographic_masses.data.txt",names=columns,na_values=["?"])

results = {}

print(df.describe())

dropped = df.loc[(df['age'].isnull()) |
              (df['shape'].isnull()) |
              (df['margin'].isnull()) |
              (df['density'].isnull())]

#удаляем все не полностью заполненные данные полагая что это шум
df.dropna(inplace=True)
#если распределение не изменилось, мы удалили рандомный шум - все хорошо.
print(df.describe())

features = df.iloc[:,[1,2,3,4]].values
predictions = df.iloc[:,5].values
predictions_labels = ["benign","malignant"]
features_labels = ["age", "shape", "margin", "density"]

#нормализуем данные
scaller = StandardScaler().fit(features)
scale_features = scaller.transform(features)

train_inputs,test_inputs,train_outputs,test_outputs = train_test_split(scale_features,predictions,train_size=.75)

#без скалинга
train_inputs_,test_inputs_,train_outputs_,test_outputs_ = train_test_split(features,predictions,train_size=.75)

###проверяем статистику
train_stat = pd.DataFrame(scaller.inverse_transform(train_inputs),columns=features_labels)
test_stat = pd.DataFrame(scaller.inverse_transform(test_inputs),columns=features_labels)

#вывод должен быть примерно одинаковым
print(train_stat.describe())
print(test_stat.describe())

#обучим дерево без скалинга для лучшей интерпритируемости
tree_model = tree.DecisionTreeClassifier().fit(train_inputs_,train_outputs_)
# dot_data = StringIO()  
# tree.export_graphviz(tree_model, out_file=dot_data,  
#                          feature_names=features_labels)  
# graph = graph_from_dot_data(dot_data.getvalue())  
# img = Image(graph.create_png()) 

# with open("./data mining/plot/final/tree.png", "wb") as png:
#     png.write(img.data)

cv_scores = cross_val_score(tree_model,test_inputs_,test_outputs_,cv=10)

results["decision tree"] = [cv_scores.mean()]

C = 1.0
svc = svm.SVC(kernel='linear', C=C)

cv_scores = cross_val_score(svc, test_inputs, test_outputs, cv=10)
cv_scores.mean()

results["svc"] = [cv_scores.mean()]

clf = neighbors.KNeighborsClassifier(n_neighbors=7)
cv_scores = cross_val_score(clf, test_inputs, test_outputs, cv=10)

results["knn"] = [cv_scores.mean()]

scaler = MinMaxScaler()
all_features_minmax = scaler.fit_transform(test_inputs)

clf = MultinomialNB()
cv_scores = cross_val_score(clf, all_features_minmax, test_outputs, cv=10)

results["baies"] = [cv_scores.mean()]

C = 1.0
svc = svm.SVC(kernel='rbf', C=C)
cv_scores = cross_val_score(svc, test_inputs, test_outputs, cv=10)
results["svc rbf"] = [cv_scores.mean()]

C = 1.0
svc = svm.SVC(kernel='sigmoid', C=C)
cv_scores = cross_val_score(svc, test_inputs, test_outputs, cv=10)
results["svc sigmoid"] = [cv_scores.mean()]

C = 1.0
svc = svm.SVC(kernel='poly', C=C)
cv_scores = cross_val_score(svc, test_inputs, test_outputs, cv=10)
results["svc poly"] = [cv_scores.mean()]

clf = LogisticRegression()
cv_scores = cross_val_score(clf, test_inputs, test_outputs, cv=10)
results["log regression"]  = [cv_scores.mean()]

# Wrap our Keras model in an estimator compatible with scikit_learn
estimator = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)
# Now we can use scikit_learn's cross_val_score to evaluate this model identically to the others
cv_scores = cross_val_score(estimator, test_inputs, test_outputs, cv=10)
results["neural network"] = [cv_scores.mean()]

print(pd.DataFrame.from_dict(results))


