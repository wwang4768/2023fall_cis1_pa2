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
        self.sample_data = np.array([[6, 6, 6], [1,1,1], [3,0,3]])
        self.source_points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.target_points = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
        self.set_registration = setRegistration()
    '''
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
    '''
    def test_calculate_3d_transformation(self):
        transformation_matrix = self.set_registration.calculate_3d_transformation(self.source_points, self.target_points)
        self.assertEqual(transformation_matrix.shape, (4, 4))

    def test_compute_error(self):
        transformation_matrix = self.set_registration.calculate_3d_transformation(self.source_points, self.target_points)
        error = self.set_registration.compute_error(self.source_points, self.target_points, transformation_matrix)
        self.assertLess(error, 1.0)

    def test_apply_transformation(self):
        transformation_matrix = self.set_registration.calculate_3d_transformation(self.source_points, self.target_points)
        transformed_points = self.set_registration.apply_transformation(self.source_points, transformation_matrix)
        self.assertEqual(transformed_points.shape, self.source_points.shape)
    
    # Test parsing
    def test_parseData(self):
        input_file = 'pa2_student_data\pa2-debug-a-calreadings.txt'
        point_cloud = parseData(input_file)
        self.assertIsInstance(point_cloud, np.ndarray)
        self.assertEqual(point_cloud.shape[1], 3)
        self.assertGreater(point_cloud.shape[0], 0)

    def test_parseCalbody(self):
        input_file = 'pa2_student_data\pa2-debug-a-calbody.txt'
        point_cloud = parseData(input_file)
        d, a, c = parseCalbody(point_cloud)
        self.assertEqual(len(d), 8)
        self.assertEqual(len(a), 8)
        self.assertEqual(len(c), 27)

    def test_parseOptpivot(self):
        input_file = 'pa2_student_data\pa2-debug-a-optpivot.txt'
        point_cloud = parseData(input_file)
        frames_d, frames_h = parseOptpivot(point_cloud, 8, 6)
        self.assertTrue(len(frames_d) > 0)
        self.assertTrue(len(frames_h) > 0)

    def test_parseFrame(self):
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        frame_chunk = 4
        frames = parseFrame(test_data, frame_chunk)
        self.assertEqual(len(frames), len(test_data) // frame_chunk)

if __name__ == '__main__':
    unittest.main()
