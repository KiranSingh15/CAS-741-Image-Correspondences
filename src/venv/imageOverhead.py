from pathlib import Path
import os

def setHeadDirPath():
    head_dir = Path(os.getcwd())  # Convert to Path object
    print(head_dir)
    return head_dir

def setInputImgPath(head_dir):
    img_dir = head_dir / "Raw_Images"  # Use '/' for safe path joining
    print(img_dir)
    return img_dir

def getInputImgNames(img_dir):
    img_dir = Path(img_dir)  # Ensure it's a Path object
    input_img = [(file.name, file.suffix) for file in img_dir.iterdir() if file.is_file()]
    return input_img

# create output paths
def createOutputDir(head_dir, dir_type):
    if dir_type == 1:   # converted greyscale images
        subfolder = "gsImagery"
    elif dir_type == 2: # Gaussian smoothed images
        subfolder = "gkImagery"
    elif dir_type == 3: # detected keypoints
        subfolder = "kpDetection"
    elif dir_type == 4: # feature descriptors
        subfolder = "fDescriptors"
    elif dir_type == 5: # feature matches
        subfolder = "fMatches"

    subpath = head_dir / subfolder  # Create the subpath
    subpath.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    print(f"Subpath created: {subpath}")
    return subpath



hd = setHeadDirPath()
id = setInputImgPath(hd)
list_im = getInputImgNames(id)
print(list_im)

# Example usage
head_dir = Path.cwd()  # Get the current working directory as Path
subpath = createOutputDir(head_dir, 1)  # Create a subfolder


