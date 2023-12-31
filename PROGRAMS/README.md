Computer Integrated Surgery 1 - Programming Assignment 2 By Zechen Xv (zxu139), Esther Wang (wwang177)

Please kindly find the file listing below, as well as more information on program structure in the report. More details are discussed in our report.

PROGRAMS Folder File Listing:
calibration_library.py: the script that all relevant frame transformation, set registration and pivot calibration methods reside

dataParsing_library.py: the script that read in input data and parses it into consumable format for the algorithm in calibration_library.py to further process

distortion_library.py: the script that handles distortion correction based on Berstein polynomials through fitting and other computation, then finally returns the corrected points

homework_1.py: the driver code for the portion that was previously implemented in programming assignment 1, which read in input data from source files stored under "pa1_student_data", performs calibration and output the result to a txt file named “pa2-{input_type}-{choose_set}-output1.txt.” Please refer to the report for more details.

homework_2.py: the driver code for the portion that programming assignment 2, which read in input data from source files stored under "pa2_student_data", performs model fitting, distortion correction, and further applies registration and calibration, then output the result to a txt file named “pa2-{input_type}-{choose_set}-output2.txt.” Please refer to the report for more details.

debug_test.py: the script that contains a series of unittest that examines the basic functionality of implementation in distortion library, calibration_library and dataParsing_library which serves our debugging purpose during the development process

error_analysis.py: the script that contains functions that evaluate the error range of the output generated by homework_1.py and homework_2.py   

Terminal Commands:
To run the driver code, please enter below in the terminal:
python homework_1.py {input_type} {choose_set} 
python homework_2.py {input_type} {choose_set}

For example, "python .\homework_2.py unknown i" 

To run the driver code, please enter below in the terminal:
python debug_test.py
python error_analysis.py {input_type} {choose_set}