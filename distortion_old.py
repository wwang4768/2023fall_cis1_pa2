import numpy as np
from scipy.special import comb

# The function scales input data within the bounds of q_min and q_max
# q_min represents the min values and q_max represents the max value for scaling 
# returns NumPy array of scaled data  
def scale_to_box(data, q_min, q_max):
    return (data - q_min) / (q_max - q_min)

# N: The degree of the Bernstein polynomial.
# k: The Bernstein polynomial parameter.
# u: The input value for the Bernstein polynomial.
def bernstein(N, k, u):
    return comb(N, k) * (1 - u) ** (N - k) * u ** k


# degree: The degree of the Bernstein polynomial.
# u: A NumPy array containing input values for the polynomial.
# Returns a NumPy array representing the F matrix constructed based on the input values.
def construct_f_matrix(degree, u):
    n_points = len(u)
    f_mat = np.zeros((n_points, (degree + 1) ** 3))

    for n in range(n_points):
        c = 0
        for i in range(degree + 1):
            for j in range(degree + 1):
                for k in range(degree + 1):
                    f_mat[n][c] = bernstein(degree, i, u[n, 0]) * bernstein(degree, j, u[n, 1]) * bernstein(degree, k, u[n, 2])
                    c += 1
    return f_mat

def fit_calibration(distorted_data, ground_truth):
    q_min = np.min(distorted_data, axis=0)
    q_max = np.max(distorted_data, axis=0)

    normalized_data = scale_to_box(distorted_data, q_min, q_max)
    F = construct_f_matrix(degree=5, u=normalized_data)

    coefficients = np.linalg.lstsq(F, ground_truth, rcond=None)[0]
    
    return q_min, q_max, coefficients

# data: A NumPy array containing the data to be corrected.
# q_min: A NumPy array representing the minimum values used for scaling.
# q_max: A NumPy array representing the maximum values used for scaling.
# coefficients: A NumPy array of coefficients obtained from calibration.
def correct_data(data, q_min, q_max, coefficients):
    normalized_data = scale_to_box(data, q_min, q_max)
    F = construct_f_matrix(degree=5, u=normalized_data)

    corrected_data = F @ coefficients
    return corrected_data

# Testing the functions
if __name__ == '__main__':
    distorted_data = np.random.rand(10000, 3) * 10
    ground_truth_data = distorted_data + np.random.randn(10000, 3) * 0.1
    sample_data = np.array([[5, 5, 5], [1, 1, 1]])

    q_min, q_max, coefficients = fit_calibration(distorted_data, ground_truth_data)
    corrected_sample = correct_data(sample_data, q_min, q_max, coefficients)
    print(corrected_sample)
