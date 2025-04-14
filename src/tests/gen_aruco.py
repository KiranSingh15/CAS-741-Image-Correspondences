from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

np.random.seed(42)  # or any other integer seed

# Output path
output_dir = Path("genARUCOImages/aruco_dataset")
output_dir.mkdir(parents=True, exist_ok=True)

# Set up ArUco dictionary and grid board
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

board = cv.aruco.GridBoard(
    size=(5, 5),  # ✅ New required format
    markerLength=100,
    markerSeparation=20,
    dictionary=aruco_dict,
)

# Draw base ArUco board
image_size = (600, 600)
base_img = cv.aruco.drawPlanarBoard(board, image_size, marginSize=10, borderBits=1)
cv.imwrite(str(output_dir / "aruco_000.png"), base_img)

# Metadata list for transforms
transform_metadata = []


# Apply random transformation
def apply_random_transform(img, i):
    rows, cols = img.shape[:2]
    angle = np.random.uniform(-45, 45)
    scale = np.random.uniform(0.5, 0.9)
    tx = np.random.randint(-30, 30)
    ty = np.random.randint(-30, 30)

    # Create affine transform
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    warped = cv.warpAffine(img, M, (cols, rows), borderValue=255)
    filename = f"aruco_{i:03d}.png"
    cv.imwrite(str(output_dir / filename), warped)

    transform_metadata.append(
        {
            "filename": filename,
            "angle_deg": angle,
            "scale": scale,
            "translate_x_px": tx,
            "translate_y_px": ty,
        }
    )


# Generate 20 transformed images
for i in range(1, 21):
    apply_random_transform(base_img, i)

# Save transformation metadata
df = pd.DataFrame(transform_metadata)
df.to_csv(output_dir / "aruco_transforms.csv", index=False)
