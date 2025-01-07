import shutil
import typer
from loguru import logger
from src.config import IMAGE_SIZE, PROCESSED_DATA_DIR, RAW_DATA_DIR
import os
import cv2
import numpy as np
from pathlib import Path
from skimage.filters import median
from skimage.morphology import disk
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks

app = typer.Typer()


def process_image(img_file: Path, input_path: Path, output_path: Path):
    """
    Processes a single image by applying preprocessing steps such as grayscale conversion,
    normalization, filtering, edge detection, circle detection, resizing, and augmentation.

    Arguments:
    - img_file : Path to the image file to process.
    - input_path : Path to the folder containing the raw images.
    - output_path : Path to the folder where processed images will be saved.
    """
    img = cv2.imread(str(img_file))

    if img is None:
        logger.error(f"Erreur : Impossible de lire l'image {img_file}")
        return  # Skip if the image cannot be read

    # Step 1: Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Normalize intensities
    normalized_img = gray_img / 255.0

    # Step 3: Remove reflections (median filter)
    filtered_img = median(normalized_img, disk(3))

    # Step 4: Edge detection (Canny)
    edges = canny(filtered_img, sigma=1.0)

    # Step 5: Circle detection (Hough transform)
    hough_radii = np.arange(10, 50, 2)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    if len(cx) > 0:  # If a circle is detected
        output_img = np.copy(filtered_img)
        cv2.circle(output_img, (cx[0], cy[0]), radii[0], (255, 255, 255), 2)
    else:
        output_img = filtered_img

    # Step 6: Resize image
    img_resized = cv2.resize(output_img, IMAGE_SIZE)

    # Step 7: Augment image
    augmented_img = augment_image(img_resized)

    # Save the processed image
    output_file = output_path / img_file.name
    success = cv2.imwrite(str(output_file), (augmented_img * 255).astype(np.uint8))

    if success:
        logger.success(f"Image sauvegardée : {output_file}")
    else:
        logger.error(f"Échec de sauvegarde : {output_file}")


def preprocess_images_training(input_dir: Path, output_dir: Path):
    """
    Preprocesses all images in the input directory and saves them to the output directory.

    Arguments:
    - input_dir : Path to the folder containing raw images.
    - output_dir : Path to the folder where processed images will be saved.
    """

    # Remove output directory if it exists and is not empty
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)  # Use shutil.rmtree to remove non-empty directories

    os.makedirs(output_dir, exist_ok=True)

    # Process images by label (normal and cataract)
    for label in ["normal", "cataract"]:
        input_path = input_dir / label
        output_path = output_dir / label
        output_path.mkdir(parents=True, exist_ok=True)

        for img_file in input_path.glob("*.*"):  # Loop through image files
            process_image(img_file, input_path, output_path)


def preprocess_images_predict(input_dir: Path, output_dir: Path):
    """
    Preprocesses all images in the input directory and saves them to the output directory.

    Arguments:
    - input_dir : Path to the folder containing raw images.
    - output_dir : Path to the folder where processed images will be saved.
    """
    # Remove output directory if it exists and is not empty
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)  # Use shutil.rmtree to remove non-empty directories

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process images by label (normal and cataract)
    for img_file in input_dir.glob("*.*"):  # Loop through image files
        process_image(img_file, input_dir, output_dir)


def augment_image(img: np.ndarray) -> np.ndarray:
    """
    Applies simple image augmentation such as rotation and zoom.

    Arguments:
    - img : Image to augment.

    Returns:
    - augmented_img : Augmented image.
    """
    # Rotation
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2), angle, 1)
    rotated_img = cv2.warpAffine(img, M, IMAGE_SIZE)

    # Zoom
    zoom_factor = np.random.uniform(0.8, 1.2)
    zoomed_img = cv2.resize(rotated_img, None, fx=zoom_factor, fy=zoom_factor)

    # Resize to target size
    augmented_img = cv2.resize(zoomed_img, IMAGE_SIZE)

    return augmented_img


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR,
):
    """
    Main function to preprocess images in training and test sets.
    """
    logger.info("Prétraitement des images...")

    # Preprocess images
    preprocess_images_training(input_path, output_path)

    logger.success("Prétraitement terminé.")


# Run the script if executed directly
if __name__ == "__main__":
    app()
