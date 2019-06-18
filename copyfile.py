# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:30:11 2019

@author: Ivory.Lu
"""
import csv
from shutil import copyfile
import pandas as pd

#data = pd.read_csv('C:/Users/ivory.lu/animalornot/summary/3000.csv')

i = 0

with open('E:/test/test.csv') as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader: 
        imageFileNames = row[2]
        Path = row[1]
        
        dst = str('E:/test' + "\\" + row[2])
        print(dst)
        
#        from shutil import copyfile
        
        copyfile(Path, dst)
        
        i = i + 1

        