#Import
import requests
from bs4 import BeautifulSoup

#get request
webpage_response = requests.get('https://s3.amazonaws.com/codecademy-content/courses/beautifulsoup/shellter.html')

#convert to soup object
webpage = webpage_response.content
soup = BeautifulSoup(webpage, "html.parser")

#print the p tag
print(soup.p)
#print the content in the p tag
print(soup.p.string)

#print all the children under the div
for child in soup.div.children:
  print(child)
  
#find all the a element
turtle_links = soup.find_all("a")
print(turtle_links)
