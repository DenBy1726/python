import pandas as pd

df = pd.read_csv("./pandas/PastHires.csv")
df.head(5)
print(df.shape)
print(df.size)
print(len(df))
print(df.columns)
print(df['Hired'][:5])
print(df['Hired'][5])
print(df.sort_values(['Years Experience', 'Hired'])[:12])
degree_counts = df['Level of Education'].value_counts()
print(degree_counts)
degree_counts.plot(kind='bar')