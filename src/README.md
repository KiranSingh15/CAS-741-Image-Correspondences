# Prerequisite Conditions
Please ensure that you have installed the software and prepared the virtual environment (venv_ifcs) before you attempt to run the software.


# File Structure
Upon initial download, your src folder should have the following file structure.
```
`src`
|   Makefile
|   README.md
|   Requirements.txt
|   projectFiles
    └── ... (assiciated core program scripts, inputs and outputs)
    .venv_ifcs
    └── ... (configuration files, dependencies, and scripts for the venv)
```
where \src\ is the custom name of your parent directory. If you have not set up the virtual environment, then you must complete the CMake file. These steps are repeated below.

---
---
# Set up Virtual Environment (venv)
## Making the Virtual Environment and Loading Dependencies
In the head `src` folder, run the following command, depending on your operating system. If you are using Windows, then be sure to use Git Bash as this follows a Unix style of shell.
In the head `src` folder, run the following command, depending on your operating system. If you are using Windows, then be sure to use Git Bash as this follows a Unix style of shell.

**Windows**
```bash
choco install make
```

**MacOS**,
```shell 
xcode-select --install
```
**Ubuntu/Linux**
```bash
sudo apt install make
```
-----------
## Results from the Make file

Once the virtual environment has been created and its dependencies are imported, you should see a new folder titled `venv_ifcs` under the main `src` folder.

``` Folder Structure
`\src\`
|   Makefile
|   README.md
|   Requirements.txt
|   projectFiles
    ├── ControlModule.py
    ├── FeatureDescriptorModule.py 
    ├── FeatureMatchingModule.py 
    ├── ImagePlotModule.py 
    ├── ImageSmoothingModule.py 
    ├── InputFormatModule.py 
    ├── KeypointDetectionModule.py 
    ├── OutputFormatModule.py 
    ├── OutputVerificationModule.py 
    ├── SpecificationParametersModule.py 
    ├── Outputs
        └── ... (folder which contains all outputs from the IFCS. Overwrites data from the previous run)
    ├── Output_Archive
    ├── Raw_Image_Lib 
        └── ... (contains generic imagery sets)
    └── Raw_Images  
|   .venv_ifcs
    ├── Include
    ├── Lib
    |    └── ... (associated Python dependencies)    
    ├── Scripts
         └── ... (associated Activate, Deactivate, and PIP files)  
    ├── Share
         └── ... (additional files, not to be altered)  
    ├── .gigignore
    └── pyvenv.cfg(associated virtual environment files)
```

where `\src\` is the custom name of your parent directory.

---
---
# Project Files - Subfolders
Navigate to the `projectFiles` subfolder in your Command Line/Shell/Bash tool. 

#### 1. Raw_Images
The `Raw_Images` folder is the most important folder in the system. The user can input a collection of JPG and PNG images to be processed by the IFCS software. We leave the specific naming scheme of the imagery up to the user as this may vary depending on their setup. However, it is recommended that the filename of every image have the characteristics as follows.
1. A unique identifier for the camera used to capture the image.
2. A unique identifier of the robot pose for which the image was captured. This may take the form of a timestamp or enumerated index. 
<br>
<br>

#### 2. Raw_Image_Lib
The `Raw_Image_Lib` folder contains some sample imagery sets for the user to familiarize themselve with the software. The imagery sets follow.
1. A Tesla Cybertruck
2. A LEGO houseplant
3. A building scene at McMaster University
4. A video game cover
<br>
<br>

#### 4. Outputs
The `Outputs` folder contains all the subfolders and data that are produced by the IFCS software. See the **Outputs Folder** section for more details.
<br>
<br>
*Note that, after the program has been run once, additional folders will be introduced. These folders follow below.*

#### 4. Output_Archive
The `Output_Archive` folder contains no data and the IFCS software does not interact with it directly. Its presence is solely to provide the user with a local staging ground for output imagery and CSV data between subsequent runs of the software.
<br>
<br>
*Note that, after the program has been run once, additional folders will be introduced. These folders follow below.*


-----------------------
-----------------------
# Running the IFCS Software

## Activate The Virtual Environment
Once the make file has been processed, you can access the virtual environment (`venv_ifcs`) by running the following command in the head project folder: 
`venv_ifcs/Scripts/activate`

## Loading Input Images
Upload the JPG and PNG images that you wish to process into the `Raw_Images` folder. 


## Configure the Processing Methods
Next, review the selected methods of operation in 
`SpecificationParametersModule.py`. Default methods are outlined below.

<br>
Activity	Status	Specific Method
Image Smoothing	Enabled	Gaussian Smoothing (Method = 1)
Keypoint Detection	Enabled	FAST – Features from Accelerated Segment Test (Method = 1)
Feature Assignment	Enabled	BRIEF – Binary Robust Independent Elementary Features (Method = 1)
Feature Matching	Enabled	Brute Force (Method = 1)
## Configure the Processing Parameters
For selected methods, the user may adjust certain parameters to improve the response of the system to inputs such as noise within the image environment. These parameters are outlined in the `InputFormatParameters.py` file.

1. Gaussian Kernel, k: Any odd integer between 3 and 15. Default = 5.
2. Noise Smoothing Standard Deviation: Any real, positive number 0 < sigma <= 10. Default = 1.
3. FAST Keypoint Pixel Intensity Threshold, t: 2 <= t <= 254.Default = 15.
4. Descriptor Bin Bound, b: 1 <= b <= 2048. Default = 2000.
5. Patch Size, p: Any positive integer between 3 and 11. Default = 31. 

## Run the IFCS Software
Once satisfied with your selection of input images, processing methods, and parameters,  run the **Control Module** by 
running the following command while inside of the `projectFiles` folder. You **MUST** be inside the `projectFiles` 
folder or else any attempt to run the Control Module will fail.

`python ControlModule.py` 

Note that you may need to specify that you are using Python3 if you are using Unix or MacOS.

`python3 ControlModule.py` 


## Program Outputs
All outputs are saved to subfolders within the parent `Outputs` folder.

```
`src`
|   Makefile
|   README.md
|   Requirements.txt
|   projectFiles
|   ├── ... (Misc. Project Files)
    └── Outputs
        └── ... (folder which contains all outputs from the IFCS. Overwrites data from the previous run)
```
#### 4. gsImagery
This folder contains greyscale images that are generated as output PNG files.
<br>
<br>

#### 5. gkImagery
This folder contains greyscale images that are generated as output PNG files.
<br>
<br>

#### 6. kpDetection
This folder contains images that are generated as output PNG files with all identified keypoints. Additionally, the folder contains a CSV for each image that contains the locations of all identified keypoints.
<br>
<br>

#### 7. fDescriptors
This folder contains images that are generated as output PNG files with all identified descriptors. Additionally, the folder contains a CSV for each image that contains the locations of all assigned descriptors.
<br>
<br>

#### 8. fMatches
This folder contains images that are generated as output PNG files with all identified matches between the features of the corresponding duo of images. Additionally, the folder contains a CSV for each image that contains the locations of all matched descriptors between each image, with corresponding identifiers for the image coordinates, descriptors, camera, and pose.
<br>
<br>

---
---
# Terminating the IFCS Software
Outputs will be outlined in the `gsImagery`, `gkImagery`, `kpDetection`, `fDescriptors`, and `fMatches`. A description of the contents of each folder is outlined in **Output_Archive** section of this document.


## Disable the Virtual Environment
When you have completed your work with the program, deactivate the venv by entering `deactivate` into the console.
