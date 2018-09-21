#issue
#https://stackoverflow.com/questions/18438997/why-is-pydot-unable-to-find-graphvizs-executables-in-windows-8
import numpy as np
import pandas as pd
from sklearn import tree
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.ensemble import RandomForestClassifier
import pydotplus

input_file = "./data mining/PastHires.csv"
df = pd.read_csv(input_file, header = 0)

print(df.head())
d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
d = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d)
print(df.head())

features = list(df.columns[:6])

y = df["Hired"]
X = df[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
img = Image(graph.create_png())
with open("./data mining/plot/forest/tree.png", "wb") as png:
    png.write(img.data)

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, y)

#Predict employment of an employed 10-year veteran
print (clf.predict([[10, 1, 4, 0, 0, 0]]))
#...and an unemployed 10-year veteran
print (clf.predict([[10, 0, 4, 0, 0, 0]]))

i_tree = 0
for tree_in_forest in clf.estimators_:
    with open("./data mining/plot/forest/forest" + str(i_tree) + ".png", 'wb') as png:
        my_file = StringIO()
        tree.export_graphviz(tree_in_forest, out_file = my_file)

        graph = pydotplus.graph_from_dot_data(my_file.getvalue())  
        img = Image(graph.create_png())
        png.write(img.data)
    i_tree = i_tree + 1