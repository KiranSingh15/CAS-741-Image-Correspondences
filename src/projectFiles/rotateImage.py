import cv2
import argparse

def rotate_image_in_place(image_path, rotation_code):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    # Rotate the image
    rotated = cv2.rotate(image, rotation_code)

    # Overwrite the original image
    cv2.imwrite(image_path, rotated)
    print(f"Rotated image saved (overwritten): {image_path}")

def get_rotation_code(degree_choice):
    rotation_map = {
        1: cv2.ROTATE_90_CLOCKWISE,
        2: cv2.ROTATE_180,
        3: cv2.ROTATE_90_COUNTERCLOCKWISE
    }
    if degree_choice not in rotation_map:
        raise ValueError("Invalid rotation option. Use 1 (90°), 2 (180°), or 3 (270°).")
    return rotation_map[degree_choice]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotate an image and overwrite it.")
    parser.add_argument("image_path", help="Path to the image to be rotated.")
    parser.add_argument(
        "-r", "--rotate",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Rotation: 1 = 90°, 2 = 180°, 3 = 270° (default: 1)"
    )
    args = parser.parse_args()

    rotation_code = get_rotation_code(args.rotate)
    rotate_image_in_place(args.image_path, rotation_code)
