import pandas as pd

#Total missing values for each feature
print(dataset.isnull().sum())

#Calculate column frequency
dataset['EndStageRenal'].value_counts()

#split the column by number
split_grade = students['grade'].str.split('(\d+)', expand=True)
