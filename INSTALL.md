Instructions for installing your software.  You should also include uninstall
instructions.

# Prerequisite Software
Before proceeding, your system must have the following software installed.
1. Git Bash (Anything from version 2.43 onwards)
2. Make (Version V4.4 or higher)
3. Python (Version 3.13.0 or higher)

If you need to install Make, then follow the steps below. It is assumed that you already have Git to Git Bash on Windows.


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
    ├── Outputs
    ├── Output_Archive
    ├── Raw_Image_Lib   (contains generic imagery sets)
    └── Raw_Images  
```
where \project\ is the custom name of your parent directory.


------------------
------------------

# Run the Makefile

| **Platform**       | **Action**                               | **Command / Notes**                                |
|--------------------|-------------------------------------------|-----------------------------------------------------|
| **Windows**        | Install `make` via Chocolatey             | `powershell`:<br>`choco install make`               |
|                    | Run Makefile (Git Bash)                   | `bash`:<br>`make install`                           |
|                    | Run Makefile (PowerShell)                 | `powershell`:<br>`make install`                     |
| **macOS**          | Install `make` via Xcode tools            | `bash`:<br>`xcode-select --install`                 |
|                    | Install `make` via Homebrew               | `bash`:<br>`brew install make`                      |
|                    | Run Makefile                              | Navigate to `src` folder:<br>`bash`:<br>`make install` |
| **Linux (Ubuntu)** | Install `make`                            | `bash`:<br>`sudo apt install make`                  |
| **Linux (Fedora)** | Install `make`                            | `bash`:<br>`sudo dnf install make`                  |
| **Linux (any)**    | Run Makefile                              | Navigate to `src` folder:<br>`bash`:<br>`make install` |



-----------------------
-----------------------
# Prepare the Virtual Environment (venv) and Load Dependencies



# Activate the Virtual Environment and Load Library Dependencies

| **Platform**     | **Action**                   | **Command / Notes**                                                        |
|------------------|------------------------------|-----------------------------------------------------------------------------|
| **Windows**      | Activate venv (PowerShell)   | Navigate to the `src` folder, then run:<br>`powershell`:<br>`.venv\Scripts\Activate.ps1` |
| **macOS**        | Activate venv (bash)         | `bash`:<br>`source .venv/bin/activate`                                     |
| **Linux (any)**  | Activate venv (bash)         | `bash`:<br>`source .venv/bin/activate`                                     |

## Installing Make
In the head `src` folder, run the following command, depending on your operating system. If you are using Windows, then be sure to use Git Bash as this follows a Unix style of shell.



## Make the Virtual Environment and Load Dependencies
In the head `project` folder, run the following command, depending on your operating system. If you are using Windows, then be sure to use Git Bash as this follows a Unix style of shell.


## Activate the venv

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

The virtual environment can be activated using the following commands, depending on your operating system:

| **Platform**                           | **Virtual Environment Activation Command** |
|----------------------------------------|--------------------------------------------|
| **Windows (PowerShell)**               | `.venv\Scripts\Activate.ps1`               |
| **Windows (CMD)**                      | `.venv\Scripts\activate.bat`               |
| **macOS / Linux / Unix (bash/zsh/sh)** | `source .venv/bin/activate`                |


# Next Steps
You have now successfully installed the IFCS software! Please read the READ.ME in the `\project\` folder. This document outlines all the remaining steps required to run the IFCS software.

# Uninstallation
uninstallation of the virtual environment and its dependencies can be achieved on all OS systems by running the following command within the `src` root. IF using windows, please use Git Bash.

``` bash
make clean
```
