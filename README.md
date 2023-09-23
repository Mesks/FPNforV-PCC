# Feature Pyramid Network based Video-based point cloud compress Fast Coding Unit Partition Approach
This is the official repository of source codes and deployment methods for the paper "FPN-based V-PCC Fast CU Partition Approach". In order to reduce its meaning and express it uniformly, the following takes "FPNforV-PCC" as the root directory, for example, the location of "README.md" is "/README.md"

<b>If you have contacted TMC2 and VTM, you can skip subsequent lengthy instructions and directly use the source under “/TMC2/dependencies” and "/TMC2/source" to change the source of original "dependencies" and "source" of TMC2 and check the methods described in our paper.  If you are not familiar with the package structure of TMC2, it is strongly recommended that you configure it as described below.</b>

## <b>Resource Link
The program versions used in the experiment are as follows (You can get their official versions through the link after quotation marks): 

1. TMC2-v18.0: https://github.com/MPEGGroup/mpeg-pcc-tmc2/tree/release-v18.0
2. HDRTools-v0.18: https://gitlab.com/standards/HDRTools/-/tree/0.18-dev
3. MPEG test sequence: https://mpeg-pcc.org/index.php/pcc-content-database/ or http://plenodb.jpeg.org/pc/8ilabs/

## <b>Content Introduction
Please note that this source code runs on Linux, if runs on Windows is necessary we strongly commend to use visual studio 2019 or later for compilation and running executable files. A brief introduction to the content provided is listed below:  

- dataExtracting: Please store the program used for extracting training data. Please use "/dataExtracting/dependencies" and "/dataExtracting/source" replace the files with the same names located in the standard "TMC2". Additionally, modify the path names from line 88 to 104 in the file located at '/dataExtracting/source/lib/PccLibCommon/source/MesksLib.cpp' to match the actual environment where the feature files will be stored. It's important to note that all modifications in dataExtracting program are defined within the 'MesksTest' macro, which can be located by searching for the corresponding string.

- FPN_training: Store the Python program for training FPN. The author's runtime environment is Python 3.10.12, PyTorch 2.0.0+cu117, and NumPy 1.24.2. Before use, please ensure to make the following modifications: 1. Modify line 424 in "/FPN_training/FPN_AttrI.py" and line 403 in "/FPN_training/FPN_GeomP.py" to change the log and model saving paths; 2. Modify lines 429 to 431 in "/FPN_training/FPN_AttrI.py" and lines 408 to 410 in "/FPN_training/FPN_GeomP.py" to adjust the dataset location.

- TMC2: Store the experimental program for simulation and verification, including the algorithm program described in the paper. Similar to the "extractingData" operation, replace the original dependencies and source with "/TMC2/dependencies" and "/TMC2/source", respectively. In addition, "/TMC2/external/" contains the driver program and models for training the network. All code modifications are defined within the "MesksTest" macro definition, which can be located by searching for that string. Before running the program, please make the following changes: 1. Modify line 305 in "/TMC2/dependencies/VTM/source/Lib/MesksLib.cpp" to use either a relative or absolute path to "/TMC2/external"; 2. Modify lines 318 and 319 in "/TMC2/dependencies/VTM/source/Lib/MesksLib.cpp" to use relative or absolute paths to "/TMC2/external/FPN_GeomP.pt" and "TMC2/external/FPN_AttrI.pt", respectively. 

## <b>Running Enviroment
In TMC2, program has dependencies on Python, ensure that your runtime environment can link to Python. For Linux environments, make sure you have a working Python environment. For Windows environments, add the 'Scripts' and 'site-packages' directories of Python to the 'Path' environment variable. Additionally, configure the following environment variables: 1. Set 'PYTHONPATH' to the path of 'pythonw.exe'; 2. Set 'PYTHONHOME' to the path of the 'Lib' directory under the directory where 'pythonw.exe' is located. Usually, they are under "C:\Users\Mesks\AppData\Roaming\Python\Python310" or Anaconda enviroment location. For example, in author's device, "Path" is added "C:\Users\Mesks\AppData\Roaming\Python\Python310\Scripts" and "c:\users\mesks\appdata\roaming\python\python310\site-packages", PYTHONHOME is "D:\Anaconda", and PYTHONPATH is "D:\Anaconda\Lib".