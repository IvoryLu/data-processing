import codecademylib3_seaborn
from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("some")
webpage_response = requests.get("https://s3.amazonaws.com/codecademy-content/courses/beautifulsoup/cacao/index.html")
webpage = webpage_response.content
soup=BeautifulSoup(webpage,"html.parser")
ratings = []
rating = soup.find_all(attrs={"class":"Rating"})
for rate in rating[1:]:
  ratings.append(float(rate.get_text()))
print(ratings)

plt.hist(ratings)
plt.show()

companies = soup.select(".Company")
all_company = []
for company in companies[1:]:
  all_company.append(company.get_text())
  
print(all_company)

data = {"Company":all_company, "Rating":ratings}
df = pd.DataFrame.from_dict(data)
mean_vals = df.groupby("Company").Rating.mean()
ten_best = mean_vals.nlargest(10)
print(ten_best)

cocoa_percents = []
cocoa_percent_tags = soup.select(".CocoaPercent")
for td in cocoa_percent_tags[1:]:
  percent = float(td.get_text().strip('%'))
  cocoa_percents.append(percent)
print(cocoa_percents)

data = {"Company":all_company, "Rating":ratings, "CocoaPercentage":cocoa_percents}
df = pd.DataFrame.from_dict(data)
plt.clf()
plt.scatter(df.CocoaPercentage, df.Rating)
z = np.polyfit(df.CocoaPercentage, df.Rating, 1)
line_function = np.poly1d(z)
plt.plot(df.CocoaPercentage, line_function(df.CocoaPercentage), "r--")
plt.show()

