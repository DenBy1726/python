import numpy as np
import pandas as pd
from sklearn import tree
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.ensemble import RandomForestClassifier
import pydotplus
import matplotlib.pyplot as plt

input_file = "./data mining/PastHires.csv"
df = pd.read_csv(input_file, header = 0)

d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
d = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d)

features = list(df.columns[:6])

y = df["Hired"]
X = df[features]

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)
# display the relative importance of each attribute


values = model.feature_importances_

plt.bar([*map(lambda x: x[:6],features)],values)
plt.savefig("./data mining/plot/featureselection/tree.png",format="png")

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(X, y)
# summarize the selection of the attributes
plt.bar([*map(lambda x: x[:6],features)],rfe.ranking_)
plt.savefig("./data mining/plot/featureselection/logregression.png",format="png")
