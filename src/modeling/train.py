# Standard Library Imports
from pathlib import Path
import pickle

# Third-Party Libraries
import typer
from loguru import logger
import pandas as pd

# import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE  # For handling imbalanced datasets

# Project Imports
from src.config import MODELS_DIR, PROCESSED_DATA_DIR

# Typer Application
app = typer.Typer()


@app.command()
def main(
    feature_cataract_path_csv: Path = PROCESSED_DATA_DIR / "feature_matrix_cataract.csv",
    feature_normal_path_csv: Path = PROCESSED_DATA_DIR / "feature_matrix_normal.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    """Main function to train and save a logistic regression model."""
    # 1. Load the datasets
    logger.info("Loading datasets...")
    normal_df = pd.read_csv(feature_normal_path_csv)
    cataract_df = pd.read_csv(feature_cataract_path_csv)

    # Add labels: 0 for normal, 1 for cataract
    normal_df["label"] = 0
    cataract_df["label"] = 1

    # Merge datasets
    df = pd.concat([normal_df, cataract_df], ignore_index=True)

    # 2. Exploratory Data Analysis (EDA)
    logger.info("Performing Exploratory Data Analysis...")
    perform_eda(df)

    # 3. Data Preparation
    logger.info("Preparing data...")
    X, y = df.iloc[:, :-1], df["label"]

    # Handle class imbalance using SMOTE
    logger.info("Applying SMOTE for class imbalance...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    # 4. Train-Test Split
    logger.info("Splitting data into training, validation, and test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 5. Model Training
    logger.info("Training the logistic regression model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    logger.info(f"Saving model to {model_path}...")
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    # Validate the model
    logger.info("Validating the model...")
    y_val_pred = model.predict(X_val)
    logger.success(classification_report(y_val, y_val_pred))

    # 6. Evaluate the Model
    logger.info("Evaluating the model on the test set...")
    evaluate_model(model, X_test, y_test)


def perform_eda(df):
    """Perform Exploratory Data Analysis (EDA) on the dataset."""
    # Visualize feature distributions
    features = df.columns[:-1]  # Exclude the 'label' column
    for feature in features:
        plt.figure(figsize=(8, 4))
        sns.histplot(data=df, x=feature, hue="label", kde=True)
        plt.title(f"Distribution of {feature}")
        plt.show()

    # Boxplots for outlier detection
    for feature in features:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x="label", y=feature)
        plt.title(f"Boxplot of {feature} by Label")
        plt.show()

    # Correlation matrix
    correlation_matrix = df[features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

    # Class distribution
    sns.countplot(data=df, x="label")
    plt.title("Class Distribution")
    plt.show()


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    # Test set predictions
    y_test_pred = model.predict(X_test)
    logger.info("Test Set Performance:")
    logger.success(classification_report(y_test, y_test_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Cataract"],
        yticklabels=["Normal", "Cataract"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve and AUC
    y_test_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    app()
