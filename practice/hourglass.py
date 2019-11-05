

import math
import os
import random
import re
import sys

# Complete the hourglassSum function below.
def hourglassSum(arr):
    print(arr)
    count = 0
    num_row = len(arr)
    num_col = len(arr[0])
    max_count = -99999
    for i, row in enumerate(arr):
        for j, ele in enumerate(row):
            if j + 2 < num_col and i + 2 < num_row:
                count = arr[i][j] + arr[i][j+1] + arr[i][j+2] + arr[i+1][j+1] + arr[i+2][j]+arr[i+2][j+1]+arr[i+2][j+2]
                print(count)
                if count > max_count:
                    max_count = count 
    return max_count

if __name__ == '__main__':
    arr = []

    result = hourglassSum(arr)

    print(str(result) + '\n')

