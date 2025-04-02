Instructions for installing your software.  You should also include uninstall
instructions.


# Prerequisite Software
Before proceeding, your system must have the following software installed.
1. Git Bash (Anything from version 2.43 onwards)
2. Make (Version V4.4 or higher)
3. Python (Version 3.13.0 or higher)

# Download the Repository
Create a folder on your file directory. Ensure that project is on a local folder to your system, rather than an instance of Google Drive or OneDrive.

Navigate to https://github.com/KiranSingh15/CAS-741-Image-Correspondences.git and clone the repository to your selected drive. you can rename the src folder to what ever you please. In this document, we shall refer to this folder as `\project\`.


## File Structure
Upon initial download, your project folder should have the following file structure.
```
\project\
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
    ├── Output_Archive
    ├── Raw_Image_Lib   (contains generic imagery sets)
    └── Raw_Images  
```
where \project\ is the custom name of your parent directory.


-----------------------
-----------------------
# Virtual Environment (venv)
## Make the Virtual Environment and Load Dependencies
In the head `project` folder, run the following command, depending on your operating system. If you are using Windows, then be sure to use Git Bash as this follows a Unix style of shell.

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

Once the virtual environment has been created and its dependencies are imported, you should see a new folder titled `venv_ifcs` under the main `project` folder.
```
project
|   Makefile
|   README.md
|   Requirements.txt
|   projectFiles
    └── ... (assiciated core program scripts, inputs and outputs)
    venv_ifcs
    └── ... (configuration files, dependencies, and scripts for the venv)
```

# Next Steps
You have now successfully installed the IFCS software! Please read the READ.ME in the `\project\` folder. This document outlines all the steps required to run the IFCS software.