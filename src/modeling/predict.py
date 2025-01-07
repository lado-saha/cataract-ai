from pathlib import Path
import pandas as pd
import typer
from loguru import logger
import pickle
from src.dataset import preprocess_images_predict  # Preprocess images for prediction
from src.features import extract_features_in_directory  # Extract features from processed images
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def predict_images(input_path: Path, image_output_path: Path, features_output_csv: Path, model_path: Path):
    """
    Process images, extract features, and make predictions using a trained model.

    Arguments:
    - input_path : Path to the input folder containing raw images to be predicted.
    - image_output_path : Path to the folder where processed images will be saved.
    - features_output_csv : Path to save the extracted features as a CSV.
    - model_path : Path to the trained model (pickle file).

    Returns:
    - A DataFrame containing filenames and their predicted classes.
    """
    # Check paths
    if not input_path.exists() or not any(input_path.iterdir()):
        raise FileNotFoundError(f"Input path '{input_path}' is empty or does not exist.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    logger.info(f"Loading model from: {model_path}")
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    logger.info(f"Preprocessing images from: {input_path}")
    preprocess_images_predict(input_path, image_output_path)

    logger.info(f"Extracting features from processed images in: {image_output_path}")
    feature_matrix, filenames = extract_features_in_directory(image_output_path, return_filenames=True)

    logger.info(f"Saving extracted features to: {features_output_csv}")
    pd.DataFrame(feature_matrix).to_csv(features_output_csv, index=False)

    logger.info("Making predictions...")
    predictions = model.predict(feature_matrix)

    # Define possible classes
    classes = ["normal", "cataract"]
    predicted_classes = [classes[pred] for pred in predictions]

    # Create a DataFrame with filenames and predictions
    results_df = pd.DataFrame({"filename": filenames, "prediction": predicted_classes})
    return results_df


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "predict",  # Folder with raw input images
    image_output_path: Path = PROCESSED_DATA_DIR / "predict",  # Folder to save processed images
    features_output_csv: Path = PROCESSED_DATA_DIR / "feature_matrix_predict.csv",  # Path to save extracted features
    model_path: Path = MODELS_DIR / "model.pkl",  # Path to the trained model
):
    """
    Main function to preprocess images, extract features, and make predictions.
    """
    try:
        logger.info("Starting prediction process...")
        results = predict_images(input_path, image_output_path, features_output_csv, model_path)

        logger.info("Prediction completed.")
        logger.success(f"Results:\n{results}")

        # Save results to a CSV file
        results_output_csv = PROCESSED_DATA_DIR / "predictions.csv"
        results.to_csv(results_output_csv, index=False)
        logger.success(f"Predictions saved to: {results_output_csv}")

    except Exception as e:
        logger.error(f"Error during prediction: {e}")


if __name__ == "__main__":
    app()
