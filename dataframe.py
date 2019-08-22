import pandas as pd

#Total missing values for each feature
print(dataset.isnull().sum())

#Calculate column frequency
dataset['EndStageRenal'].value_counts()
