from matplotlib import pyplot as plt
from script import sales_times1
from script import sales_times2
# normed=True This command divides the height of each column by
# a constant such that the total shaded area of the histogram sums
# to 1 
plt.hist(sales_times1, bins=20, alpha=0.4, normed=True)
plt.hist(sales_times2, bins=20, alpha=0.4, normed=True)

plt.show()
#%%
from matplotlib import pyplot as plt

exam_scores1 = [62.58, 67.63, 81.37, 52.53, 62.98, 72.15, 59.05, 73.85, 97.24, 76.81, 89.34, 74.44, 68.52, 85.13, 90.75, 70.29, 75.62, 85.38, 77.82, 98.31, 79.08, 61.72, 71.33, 80.77, 80.31, 78.16, 61.15, 64.99, 72.67, 78.94]
exam_scores2 = [72.38, 71.28, 79.24, 83.86, 84.42, 79.38, 75.51, 76.63, 81.48,78.81,79.23,74.38,79.27,81.07,75.42,90.35,82.93,86.74,81.33,95.1,86.57,83.66,85.58,81.87,92.14,72.15,91.64,74.21,89.04,76.54,81.9,96.5,80.05,74.77,72.26,73.23,92.6,66.22,70.09,77.2]

# Make your plot here
plt.figure(figsize=(10,8))
plt.hist(exam_scores1,bins=12,normed=True,
         histtype='step',linewidth=2)
plt.hist(exam_scores2,bins=12,normed=True,
         histtype='step',linewidth=2)
legends=["1st Yr Teaching","2nd Yr Teaching"]
plt.legend(legends)
plt.title("Final Exam Score Distribution")
plt.xlabel("Percentage")
plt.ylabel("Frequency")

plt.savefig("my_histogram.png")
#%%
import numpy as np
import pandas as pd

# Import matplotlib pyplot
from matplotlib import pyplot as plt

# Read in transactions data
greatest_books = pd.read_csv("top-hundred-books.csv")

# Save transaction times to a separate numpy array
author_ages = greatest_books['Ages']

# Use numpy to calculate the average age of the top 100 authors
average_age = np.average(author_ages)

print("The average age of the 100 greatest authors, according to Le Monde is: " + str(average_age))

# Plot the figure
plt.hist(author_ages, range=(10, 80), bins=14,  edgecolor='black')
plt.title("Age of Top 100 Novel Authors at Publication")
plt.xlabel("Publication Age")
plt.ylabel("Count")
plt.axvline(average_age, color='r', linestyle='solid', linewidth=2, label="Mean")
plt.legend()

plt.show()
