# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:22:57 2019

@author: Ivory.Lu
"""
# Merge two csv files into one file
import csv
import pandas as pd

data = pd.read_csv('E:/test/mornington_400.csv')

test = pd.read_csv('E:/test/mornington_400_test.csv')

temp = pd.read_csv('C:/Users/ivory.lu/animalornot/summary/test_sequence_mornington.csv')

species = pd.read_csv('C:/Users/ivory.lu/animalornot/summary/species.csv')
#rest = []

name_org = set(data.name)
#name_test = set(test.name)
#rest = list(name_org - name_test)

#Full outer join
de_col = pd.merge(data, test, on='name', how = 'outer')
all_col = pd.merge(de_col, species, on='name', how = 'outer')

# Compare files in different folders

from os import listdir
from os.path import isfile, join
middle = set( [f for f in listdir('D:/middle') if isfile(join('D:/middle', f))])
test = set([f for f in listdir('D:/test') if isfile(join('D:/test', f))])                 
half = set([f for f in listdir('D:/half') if isfile(join('D:/half', f))])

rest = list(name_org - middle -test - half)
