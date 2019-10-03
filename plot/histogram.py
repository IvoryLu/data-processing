from matplotlib import pyplot as plt
from script import sales_times1
from script import sales_times2
# normed=True This command divides the height of each column by
# a constant such that the total shaded area of the histogram sums
# to 1 
plt.hist(sales_times1, bins=20, alpha=0.4, normed=True)
plt.hist(sales_times2, bins=20, alpha=0.4, normed=True)

plt.show()
