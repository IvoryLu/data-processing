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

#ANOVA 
#It tests the null hypothesis that all of the datasets have the same mean.
#If we reject the null hypothesis (small p-value) with ANOVA, we're saying that 
#at least one of the sets has a different mean;but we can't make any conclusions on
#which two populations have a significant difference

fstat, pval = f_oneway(a,b,c)
print(pval)

#Tukey's Range Test
#Tukey's Test can tell us which pairs of locations are distinguishable from each other. 
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
import numpy as np

a = np.genfromtxt("store_a.csv",  delimiter=",")
b = np.genfromtxt("store_b.csv",  delimiter=",")
c = np.genfromtxt("store_c.csv",  delimiter=",")

stat, pval = f_oneway(a, b, c)
print pval

# Using our data from ANOVA, we create v and l
v = np.concatenate([a, b, c])
labels = ['a'] * len(a) + ['b'] * len(b) + ['c'] * len(c)

tukey_results = pairwise_tukeyhsd(v,labels,0.05)

print(tukey_results)

