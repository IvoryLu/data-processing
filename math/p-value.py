from scipy.stats import ttest_1samp
import numpy as np

for i in range(1000): # 1000 experiments
   #Test One Sample T-test
   #Expected average is 30. 
   tstatistic, pval = ttest_1samp(daily_visitors[i], 30)
   #print the pvalue here:
   print pval

# P-value give us an idea of how confident we can be in a result. 
