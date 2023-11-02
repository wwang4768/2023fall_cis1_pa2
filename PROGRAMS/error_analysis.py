import numpy as np, copy, os, re, argparse
from calibration_library import *
from dataParsing_library import *
from validation_test import *
from distortion_library import * 

def calculate_error_from_sample(self, file1, file2, use_reference=0):
    with open(file1, 'r') as f1:
        next(f1)
        data1 = [list(map(float, line.strip().split(','))) for line in f1]

    with open(file2, 'r') as f2:
        next(f2)
        data2 = [list(map(float, line.strip().split(','))) for line in f2]

    reference_data = data1 if use_reference == 0 else data2

    percentage_differences = []
    for row1, row2 in zip(data1, data2):
        percentage_diff_row = []
        for val1, val2 in zip(row1, row2):
            percentage_diff = ((val2 - val1) / val1)
            percentage_diff_row.append(percentage_diff)
        percentage_differences.append(percentage_diff_row)

    return percentage_differences

def main():
    calculate_error_from_sample()

if __name__ == '__main__':
    main()
