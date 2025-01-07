import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

# Inbuilt imports
import typer
from loguru import logger

# from tqdm import tqdm

from src.config import PROCESSED_DATA_DIR

app = typer.Typer()


def extract_top_circle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=100,
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        largest_circle = sorted(circles, key=lambda x: x[2], reverse=True)[0]
        return largest_circle
    return None


# 2. Extraction des caractéristiques des cercles et statistiques


def extract_circle_stats(image, circle):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if circle is not None:
        x, y, r = circle
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        masked_region = cv2.bitwise_and(gray, gray, mask=mask)

        region_mean = np.mean(masked_region[masked_region > 0])
        region_std = np.std(masked_region[masked_region > 0])
    else:
        region_mean = np.mean(gray)
        region_std = np.std(gray)

    return region_mean, region_std


# 3. Extraction des contours et densité


def extract_contour_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    return density


# 4. Normalisation des pixels


def compute_normalization_stats(image):
    normalized_image = image / 255.0
    mean_norm = np.mean(normalized_image)
    std_norm = np.std(normalized_image)
    return mean_norm, std_norm


# 5. Génération du vecteur de caractéristiques pour une image


def generate_feature_vector(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    top_circle = extract_top_circle(image)
    avg_blue = np.mean(image[:, :, 0])
    avg_green = np.mean(image[:, :, 1])
    avg_red = np.mean(image[:, :, 2])

    # Moyenne et écart type des intensités des niveaux de gris
    gray_mean, gray_std = extract_circle_stats(image, top_circle)

    # Densité des contours
    contour_density = extract_contour_density(image)

    # Moyenne et écart type des pixels normalisés
    norm_mean, norm_std = compute_normalization_stats(image)

    if top_circle is not None:
        x, y, r = top_circle
        feature_vector = [
            x,
            y,
            r,
            avg_blue,
            avg_green,
            avg_red,
            gray_mean,
            gray_std,
            contour_density,
            norm_mean,
            norm_std,
        ]
        return feature_vector

    return None


# 6. Traitement de toutes les images d'un répertoire


def extract_features_in_directory(image_dir, return_filenames=False):
    """
    Extract features from all images in a given directory.

    Args:
    - image_dir: Path to the directory containing images.

    Returns:
    - feature_matrix: A NumPy array of feature vectors.
    - filenames: A list of filenames corresponding to the feature vectors.
    """
    feature_matrix = []
    filenames = []

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)

        # Process only image files
        if image_path.endswith((".png", ".jpg", ".jpeg")):
            logger.info(f"Processing {image_name}...")

            # Generate the feature vector for the current image
            feature_vector = generate_feature_vector(image_path)

            if feature_vector is not None:
                feature_matrix.append(feature_vector)
                filenames.append(image_name)  # Keep track of the filename

    if return_filenames:
        return np.array(feature_matrix), filenames
    else:
        return np.array(feature_matrix)
# 7. Affichage et sauvegarde de la matrice des caractéristiques


def display_and_save_feature_matrix(feature_matrix, output_csv_path):
    if feature_matrix.size == 0:
        logger.error("Aucune caractéristique extraite. Vérifiez vos images ou le chemin d'accès.")
        return

    column_names = [
        "Center X",
        "Center Y",
        "Radius",
        "Avg Blue",
        "Avg Green",
        "Avg Red",
        "Gray Mean",
        "Gray Std",
        "Contour Density",
        "Norm Mean",
        "Norm Std",
    ]
    feature_df = pd.DataFrame(feature_matrix, columns=column_names)
    logger.info("Feature Matrix:")
    logger.info(feature_df)

    # Sauvegarde dans un fichier CSV
    feature_df.to_csv(output_csv_path, index=False)
    logger.success(f"Matrice des caractéristiques sauvegardée dans {output_csv_path}.")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path_cataract: Path = PROCESSED_DATA_DIR / "cataract",
    input_path_normal: Path = PROCESSED_DATA_DIR / "normal",
    output_path_cataract_csv: Path = PROCESSED_DATA_DIR / "feature_matrix_cataract.csv",
    output_path_normal_csv: Path = PROCESSED_DATA_DIR / "feature_matrix_normal.csv",
    # -----------------------------------------
):
    # Processing cataract
    feature_matrix = extract_features_in_directory(input_path_cataract)
    display_and_save_feature_matrix(feature_matrix, output_path_cataract_csv)

    # Processing normal
    feature_matrix = extract_features_in_directory(input_path_normal)
    display_and_save_feature_matrix(feature_matrix, output_path_normal_csv)


if __name__ == "__main__":
    app()
