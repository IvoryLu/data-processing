import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# import data data is currently not avaiable 
flags = pd.read_csv("flags.csv", header = 0)

# Print out the columns name and top 5 rows to get an idea about the data
print(flags.columns)
print(flags.head())

# Labels 
labels = flags[["Landmass"]]

# Select input columns 
data = flags[["Red", "Green", "Blue", "Gold",
 "White", "Black", "Orange",
 "Circles",
"Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle"]]

# split the data set
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

# Iterate with different depth of tree to get the best performance
scores = []

for i in range(1, 20):
  tree = DecisionTreeClassifier(random_state = 1, max_depth = i)
  tree.fit(train_data, train_labels)
  scores.append(tree.score(test_data, test_labels))

# Visulize the max_depth and tree performance
plt.plot(range(1,20), scores)
plt.show()

