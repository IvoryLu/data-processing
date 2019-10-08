import pandas as pd
from matplotlib import pyplot as plt

gradebook = pd.read_csv("gradebook.csv")

import numpy as np

print(gradebook)

assignment1 = gradebook[gradebook.assignment_name == 'Assignment 1']

print(assignment1)

asn1_median = np.median(assignment1.grade)

print(asn1_median)

#%%
# Import packages
import codecademylib
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

median_age = np.median(author_ages)

# Plot the figure
plt.hist(author_ages, range=(10, 80), bins=14,  edgecolor='black')
plt.title("Age of Top 100 Novel Authors at Publication")
plt.xlabel("Publication Age")
plt.ylabel("Count")
plt.axvline(average_age, color='r', linestyle='solid', linewidth=2, label="Mean")
plt.axvline(median_age, color='y', linestyle='solid', linewidth=2, label="Median")
plt.legend()

plt.show()
