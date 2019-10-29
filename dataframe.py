import pandas as pd

#Total missing values for each feature
print(dataset.isnull().sum())

#Calculate column frequency
dataset['EndStageRenal'].value_counts()

# Delete the % in the column
students['score'].replace('[\%,]','',regex=True)

#split the column by number
split_grade = students['grade'].str.split('(\d+)', expand=True)

#Import multiple files and concatanate into one dataframe
import glob
files = glob.glob(path + "/*.csv")
#files = glob.glob("file*.csv")

df_list = []
for filename in files:
  data = pd.read_csv(filename)
  df_list.append(data)
#frame = pd.concat(li, axis=0, ignore_index=True)
df = pd.concat(df_list)

print(df.columns)
print(df.dtypes)
