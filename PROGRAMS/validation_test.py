import unittest
import numpy 
from calibration_library import *
from dataParsing_library import *
from distortion_library import *

class TestDistortionCorrection(unittest.TestCase):

    def setUp(self):
        # Initialize common test data here
        np.random.seed(0) 
        self.distorted_data = np.random.rand(10000, 3) * 10
        self.ground_truth_data = self.distorted_data + np.random.randn(10000, 3) * 0.1
        self.sample_data = np.array([[6, 6, 6], [1,1,1], [3,3,3]])

    def test_calibration_and_correction(self):
        calibrator_corrected = DewarpingCalibrationCorrected()
        calibrator_corrected.fit(self.distorted_data, self.ground_truth_data)
        corrected_sample = calibrator_corrected.correction(self.sample_data)

        # You can add assertions to verify that the corrected_sample is as expected
        self.assertTrue(np.allclose(corrected_sample, self.sample_data, rtol=1e-1, atol=1e-1))

    def test_fit(self):
        # Test the fit method
        calibrator = DewarpingCalibrationCorrected()
        calibrator.fit(self.distorted_data, self.ground_truth_data)
        coefficients = calibrator.coefficients
        q_min = calibrator.q_min
        q_max = calibrator.q_max

        # Assert that coefficients, q_min, and q_max are not None
        self.assertIsNotNone(coefficients)
        self.assertIsNotNone(q_min)
        self.assertIsNotNone(q_max)

    def test_correction(self):
        # Test the correction method
        calibrator = DewarpingCalibrationCorrected()
        calibrator.fit(self.distorted_data, self.ground_truth_data)
        corrected_sample = calibrator.correction(self.sample_data)

        # Assert that the corrected_sample has the expected shape
        self.assertEqual(corrected_sample.shape, self.sample_data.shape)

        # Assert that the corrected_sample is not None
        self.assertIsNotNone(corrected_sample)

if __name__ == '__main__':
    unittest.main()
