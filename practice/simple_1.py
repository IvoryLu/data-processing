# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:24:03 2019

@author: Ivory.Lu
"""
# Complete the jumpingOnClouds function below.
def jumpingOnClouds(c):
    count = 0
    one = 1
    i = 0
    while i + 1 < len(c):
        print(i)
        if one not in c:
            count = int(len(c)/2)
            break
        if i+2 < len(c) and c[i + 2] == 0:
            i = i + 1
            count = count + 1
        if i+2 < len(c) and c[i + 2] == 1:
            count = count + 1
        if c[i] ==0 and i + 2 == len(c):
            count = count + 1 
            break
        i = i + 1
    return count

if __name__ == '__main__':

#    result = jumpingOnClouds([0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
    0 0 0 1 0 0
    print(str(result) + '\n')
    
