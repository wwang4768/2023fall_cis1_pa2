import numpy as np
from calibration_library import *
from dataParsing_library import *
from validation_test import *
from distortion_library import *
import copy
import os
import re

def main(): 
    # Read in input dataset
    script_directory = os.path.dirname(__file__)
    dirname = os.path.dirname(script_directory)
    choose_set = 'f'
    #base_path = os.path.join(dirname, 'pa2_student_data\\PA2 Student Data\\pa2-unknown-')
    base_path = os.path.join(dirname, 'pa2_student_data\\PA2 Student Data\\pa2-debug-')

    calbody = base_path + choose_set + '-calbody.txt'
    calbody_point_cloud = parseData(calbody)
    d0, a0, c0 = parseCalbody(calbody_point_cloud)

    calreading = base_path + choose_set + '-calreadings.txt'
    calreading_point_cloud = parseData(calreading)
    calreading_frames = parseFrame(calreading_point_cloud, 8+8+27) # 8 optical markers on calibration object and 27 EM markers on calibration object
    
    # new inputs to be incorporated 
    ct_fid = base_path + choose_set + '-ct-fiducials.txt'
    ct_fid_point_cloud = parseData(ct_fid)

    em_fid = base_path + choose_set + '-em-fiducialss.txt'
    em_fid_point_cloud = parseData(em_fid)
    em_fid_frames = parseFrame(em_fid_point_cloud, 6) # stores the list of 6 frames, each of which contains data of 6 EM markers on probe 


    em_nav = base_path + choose_set + '-EM-nav.txt'
    em_nav_point_cloud = parseData(em_nav)
    em_nav_frames = parseFrame(em_nav_point_cloud, 6) # stores the list of 4 frames, each of which contains data of 6 EM markers on probe 

    empivot = base_path + choose_set + '-empivot.txt'
    empivot_point_cloud = parseData(empivot)
    empivot_frames = parseFrame(empivot_point_cloud, 6) # stores the list of 12 frames, each of which contains data of 6 EM markers on probe 
    
    optpivot = base_path + choose_set + '-optpivot.txt'
    optpivot_point_cloud = parseData(optpivot) # stores the list of 12 frames, each of which contains data of 8 optical markers on EM base & 6 EM markers on probe
    optpivot_em_frames, optpivot_opt_frames = parseOptpivot(optpivot_point_cloud, 8, 6) 
    

    registration = setRegistration()
    np.set_printoptions(formatter={'float': '{:.2f}'.format})

    # Step 1
    # Use transformed Ci expected from ci, compared against real Ci
    # in calreading to calculate degree of distortion

    source_points_d = d0
    trans_matrix_d = []
    target_points = []

    for i in range(125):
        target_points = calreading_frames[i][:8]
        transformation_matrix = registration.calculate_3d_transformation(source_points_d, target_points)
        trans_matrix_d.append(transformation_matrix)

    source_points_a = a0
    trans_matrix_a = []
    target_points = []

    for i in range(125):
        target_points = calreading_frames[i][8:16]
        transformation_matrix = registration.calculate_3d_transformation(source_points_a, target_points)
        trans_matrix_a.append(transformation_matrix)
    
    source_points_c = c0
    transformation_matrix = []
    transformed_point = []
    distorted_data = []

    for i in range(125):
        distorted_data.append(calreading_frames[i][16:43])
        transformation_matrix = np.dot(np.linalg.inv(trans_matrix_d[i]), trans_matrix_a[i])
        transformed_point.append(registration.apply_transformation(source_points_c, transformation_matrix))
    
    # 27 * 125 frames Ci expected 
    # print(transformed_point)

    # Step 2
    # undistort empivot_frames
    # distorted_data
    ground_truth_data = transformed_point
    # print(len(distorted_data), len(ground_truth_data))
    # print(len(distorted_data[0]), len(ground_truth_data[0]))
    calibrator_corrected = DewarpingCalibrationCorrected()
    sample_data = empivot_frames

    distorted_data_comb = []
    ground_truth_data_comb = []

    for i in range(125):
        for j in range(27):
            distorted_data_comb.append(distorted_data[i][j])
            ground_truth_data_comb.append(ground_truth_data[i][j])
    
    #print(len(distorted_data_comb))

    # 125 frames, each of which has 27 points 
    calibrator_corrected.fit(distorted_data_comb, ground_truth_data_comb)

    # 12 frames, each of which has 6 points
    # Everything related G has to be corrected to dewarp the distortion
    # corrected_sample = calibrator_corrected.correction(sample_data)

    corrected_sample = []
    for i in range(12):
        corrected_sample.append(calibrator_corrected.correction(sample_data[i]))
    #print(sample_data[0])
    #print(corrected_sample[0])

    # Step 3
    empivot_frames = corrected_sample
    # Initalize the set for gj = Gj - G0
    translated_points_Gj = copy.deepcopy(empivot_frames)
    # Find centroid of Gj (the original position of 6 EM markers on the probe)
    midpoint = np.mean(empivot_frames, axis=1)
    trans_matrix_FG = []

    for i in range(12):
        for j in range(6):
            p = empivot_frames[i][j] - midpoint[i]
            translated_points_Gj[i][j] = p
    
    # fix gj as the original starting positions
    source_points = translated_points_Gj[0]
    for i in range(12):
        target_points = empivot_frames[i]
        transformation_matrix = registration.calculate_3d_transformation(source_points, target_points)
        trans_matrix_FG.append(transformation_matrix)
    p_tip_G, p_dimple = registration.pivot_calibration(trans_matrix_FG)
    print(p_dimple)

    # Step 4
    # Initalize the set for gj = Gj - G0
    translated_points_Gj_fid = copy.deepcopy(em_fid_frames)
    # Find centroid of Gj (the original position of 6 EM markers on the probe)
    midpoint_fid = np.mean(em_fid_frames, axis=1)
    trans_matrix_FG_fid = []

    for i in range(6):
        for j in range(6):
            p = em_fid_frames[i][j] - midpoint_fid[i]
            translated_points_Gj_fid[i][j] = p
    
    # fix gj as the original starting positions
    source_points_fid = translated_points_Gj_fid[0]
    for i in range(6):
        target_points_fid = em_fid_frames[i]
        transformation_matrix_fid = registration.calculate_3d_transformation(source_points_fid, target_points_fid)
        trans_matrix_FG_fid.append(transformation_matrix_fid)
    #print(len(trans_matrix_FG_fid))
    #print(trans_matrix_FG_fid[0])

    pivot_Bj = []
    # may have to loop through - apply transformation matrices to p_pivot_G
    for i in range(6):
        pivot_Bj.append(registration.apply_transformation_single_pt(p_dimple, trans_matrix_FG_fid[i]))
    #print(pivot_Bj)
    

    # Step 5
    source_points_bj = ct_fid_point_cloud #bj 
    target_points_bj = pivot_Bj
    transformation_matrix_Bj = registration.calculate_3d_transformation(source_points_bj, target_points_bj)
    #print(transformation_matrix_Bj)

    # Step 6
    # Initalize the set for gj = Gj - G0
    translated_points_Gj_nav = copy.deepcopy(em_nav_frames)
    #print(len(translated_points_Gj_nav))
    #print(translated_points_Gj_nav[0])
    # Find centroid of Gj (the original position of 6 EM markers on the probe)
    midpoint_nav = np.mean(em_nav_frames, axis=1)
    trans_matrix_FG_nav = []

    for i in range(4):
        for j in range(6):
            p = em_nav_frames[i][j] - midpoint_nav[i]
            translated_points_Gj_nav[i][j] = p
    
    # fix gj as the original starting positions
    source_points_nav = translated_points_Gj_nav[0]
    for i in range(4):
        target_points_nav = em_nav_frames[i]
        transformation_matrix_nav = registration.calculate_3d_transformation(source_points_nav, target_points_nav)
        trans_matrix_FG_nav.append(transformation_matrix_nav)
    # p_dimple * FGi * Freg 
    #print(trans_matrix_FG_nav)
    #print(len(transformation_matrix_Bj))

    output = []
    for i in range(4):
        trans_FG = registration.apply_transformation_single_pt(p_dimple, trans_matrix_FG_nav[i])
        output.append(trans_FG)
    
    for i in range(4):
        output[i] = registration.apply_transformation_single_pt(output[i], transformation_matrix_Bj)
    print(output)
'''
    # Q6ac
    # Initalize the set for hj = Hj - H0
    translated_points = copy.deepcopy(optpivot_opt_frames)
    H_prime = []
    # Find centroid of Hj (the original position of 6 EM markers on the probe)
    midpoint = np.mean(optpivot_opt_frames, axis=1)
    # Find transformation matrix for 12 frames
    trans_matrix_f = []

    # Calculate Fd
    source_points_d = d0 #optpivot_em_frames[0]
    transformation_matrix_Fd = []
    target_points = []

    for i in range(12):
        target_points = optpivot_em_frames[i]
        transformation_matrix = registration.calculate_3d_transformation(source_points_d, target_points)
        transformation_matrix = np.linalg.inv(transformation_matrix)
        transformation_matrix_Fd.append(transformation_matrix)

    for i in range(12):
        for j in range(6):
        # fill out hj 
            p = optpivot_opt_frames[i][j] - midpoint[i]
            translated_points[i][j] = p
    
    #apply Fd to H
    for i in range(12):
        chunk_array = np.vstack(optpivot_opt_frames[i])
        transformed_chunk = registration.apply_transformation(chunk_array, transformation_matrix_Fd[i])
        H_prime.append(transformed_chunk)

    # fix Hj as the original starting positions
    source_points = translated_points[0]

    for i in range(12):
        target_points = H_prime[i]
        transformation_matrix = registration.calculate_3d_transformation(source_points, target_points)
        trans_matrix_f.append(transformation_matrix)
    p_tip_H, p_pivot_H = registration.pivot_calibration(trans_matrix_f)

    


    # Output 2
    output_name_ct = 'pa2-unknown-' + choose_set + '-output2.txt'

    # write to output
    with open(output_name_cal, "w") as file:
        file.write('27, 8, ' + output_name_cal + '\n')
        file.write(output_string)
'''
    
if __name__ == "__main__":
    main()
    
    # v = validate()
    # file1 = 'pa1_student_data\PA1 Student Data\pa1-debug-g-output1.txt'
    # file2 = 'OUTPUT\pa1-unknown-g-output.txt'
    
    # percentage_differences = v.calculate_error_from_sample(file1, file2, use_reference=0)
    # print(np.mean(percentage_differences))
    
    