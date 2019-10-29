import pandas as pd

#Total missing values for each feature
print(dataset.isnull().sum())

#Calculate column frequency
dataset['EndStageRenal'].value_counts()

#split the column by number
split_grade = students['grade'].str.split('(\d+)', expand=True)

#Import multiple files and concatanate into one dataframe
import glob
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
