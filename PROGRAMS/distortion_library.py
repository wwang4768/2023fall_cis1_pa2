import numpy as np
from scipy.special import comb

# Create an instance of the DewarpingCalibrationCorrected class, fit it to our data, and correct a sample point
class DewarpingCalibrationCorrected:

    def __init__(self, degree=5):
        self.degree = degree
        self.coefficients = None
        self.q_min = None
        self.q_max = None

# The function scales input data within the bounds of q_min and q_max
# q_min represents the min values and q_max represents the max value for scaling 
# returns NumPy array of scaled data  
    @staticmethod
    def scale_to_box(data, q_min, q_max):
        return (data - q_min) / (q_max - q_min)

# N: The degree of the Bernstein polynomial.
# k: The Bernstein polynomial parameter.
# u: The input value for the Bernstein polynomial.
    @staticmethod
    def bernstein(N, k, u):
        return comb(N, k) * (1 - u) ** (N - k) * u ** k

# degree: The degree of the Bernstein polynomial.
# u: A NumPy array containing input values for the polynomial.
# Returns a NumPy array representing the F matrix constructed based on the input values.
    def construct_f_matrix(self, u):
        n_points = len(u)
        f_mat = np.zeros([n_points, (self.degree + 1) ** 3])

        for n in range(n_points):
            c = 0
            for i in range(self.degree + 1):
                for j in range(self.degree + 1):
                    for k in range(self.degree + 1):
                        f_mat[n][c] = self.bernstein(self.degree, i, u[n][0]) * \
                                      self.bernstein(self.degree, j, u[n][1]) * \
                                      self.bernstein(self.degree, k, u[n][2])
                        c += 1
        return f_mat

    def fit(self, distorted_data, ground_truth):
        self.q_min = np.min(distorted_data, axis=0)
        self.q_max = np.max(distorted_data, axis=0)

        normalized_data = DewarpingCalibrationCorrected.scale_to_box(distorted_data, self.q_min, self.q_max)
        F = self.construct_f_matrix(normalized_data)

        self.coefficients = np.linalg.lstsq(F, ground_truth, rcond=None)[0]

# data: A NumPy array containing the data to be corrected.
# q_min: A NumPy array representing the minimum values used for scaling.
# q_max: A NumPy array representing the maximum values used for scaling.
# coefficients: A NumPy array of coefficients obtained from calibration.
    def correction(self, data):
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet.")

        normalized_data = DewarpingCalibrationCorrected.scale_to_box(data, self.q_min, self.q_max)
        F = self.construct_f_matrix(normalized_data)

        corrected_data = F @ self.coefficients
        return corrected_data



