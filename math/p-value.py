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

#Binomial Test
# A Binomial Test compares a categorical dataset to some expectation. 
# The null hypothesis would be there is no difference between the observed behavior and 
# the expected behaviros. If we get a p-value of less than 0.05, we can reject that hypothesis. 
pval = binom_test(525, n=1000, p=0.5) #525 actual number, n total number, p expected value

#Chi Square Test
#Used for two or more categorical datasets that we want to compare.
#Men and women were both given a survey asking “Which of the following 
#three products is your favorite?” 
#Did the men and women have significantly different preferences?
from scipy.stats import chi2_contingency

# Contingency table
#         harvester |  leaf cutter
# ----+------------------+------------
# 1st gr | 30       |  10
# 2nd gr | 35       |  5
# 3rd gr | 28       |  12

X = [[30, 10],
     [35, 5],
     [28, 12],
     [20, 20]]
chi2, pval, dof, expected = chi2_contingency(X)
print pval


