import numpy as np, copy, os, re, argparse
from calibration_library import *
from dataParsing_library import *
from debug_test import *
from distortion_library import * 

def read_coordinates(file_path):
    """
    Reads a file, skips the header, and extracts coordinates, assuming they are comma-separated.

    Parameters:
    - file_path: Path to the file containing the (x, y, z) points.
    Returns:
    - A list of tuples, where each tuple contains the (x, y, z) coordinates.
    """

    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)
        # Extract and return the coordinates
        return [tuple(map(float, line.strip().split(','))) for line in file]

def compute_error(computed_data, target_data):
    """
    Computes the Euclidean distance between computed data points and target data points.

    Parameters:
    - computed_data: A list of tuples/lists representing the computed (x, y, z) points.
    - target_data: A list of tuples/lists representing the target (x, y, z) points.

    Returns:
    - A list of Euclidean distances representing the error for each corresponding point.
    """

    if len(computed_data) != len(target_data):
        raise ValueError("The number of computed data points must match the number of target data points.")

    # Calculate the Euclidean distance for each pair of points
    errors = [np.linalg.norm(np.array(computed) - np.array(target)) for computed, target in
              zip(computed_data, target_data)]
    print(errors)
    return errors


def main():
    # User interface prompt that takes input from user
    parser = argparse.ArgumentParser(description='error analysis input')
    parser.add_argument('input_type', help='The debug or unknown input data to process')
    parser.add_argument('choose_set', help='The alphabetical index of the data set')
    args = parser.parse_args()

    # Read in input dataset
    script_directory = os.path.dirname(__file__)
    dirname = os.path.dirname(script_directory)
    base_path_output = os.path.join(dirname, f'OUTPUT\\pa2-{args.input_type}-{args.choose_set}-output2.txt')
    base_path_sample = os.path.join(dirname, f'PROGRAMS\\pa2_student_data\\pa2-{args.input_type}-{args.choose_set}-output2.txt')
    computed_data = read_coordinates(base_path_output)
    target_data = read_coordinates(base_path_sample)
    error = compute_error(computed_data, target_data)

    print(error)

if __name__ == '__main__':
    main()

"""
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
"""
