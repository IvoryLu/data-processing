from scipy.stats import ttest_1samp
import numpy as np

#One Sample T-test
for i in range(1000): # 1000 experiments
   
   #Expected average is 30. 
   tstatistic, pval = ttest_1samp(daily_visitors[i], 30)
   #print the pvalue here:
   print pval

# P-value give us an idea of how confident we can be in a result. 

#Two Sample T-test
from scipy.stats import ttest_ind
import numpy as np

week1 = np.genfromtxt("week1.csv",  delimiter=",")
week2 = np.genfromtxt("week2.csv",  delimiter=",")

week1_mean = np.mean(week1)
week2_mean = np.mean(week2)
print(week1_mean)
print(week2_mean)

week1_std = np.std(week1)
week2_std = np.std(week2)
print(week2_std)
print(week1_std)

tstatistic, pval = ttest_ind(week1, week2)
print(pval)
