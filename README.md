# Cataract Detection using Traditional Machine Learning and Feature Extraction

This repository implements a machine learning pipeline to detect cataracts in eye images using traditional ML algorithms and feature extraction techniques.

## Table of Contents

1. [Project Setup](#1-project-setup)
2. [Dataset Information](#2-dataset-information)
3. [Download and Prepare the Dataset](#3-download-and-prepare-the-dataset)
4. [Training the Model](#4-training-the-model)
5. [Testing and Inference](#5-testing-and-inference)
6. [Exploratory Data Analysis (EDA)](#6-exploratory-data-analysis-eda)

## 1. Project Setup

This project is built using Python 3.10 and requires several Python dependencies. To set up your environment, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/lado-saha/cataract-ai.git
```

This command clones the project repository from GitHub to your local machine. Replace the URL with your repository link if necessary.

```bash
cd cataract-ai
```

After cloning the repository, navigate to the project directory using this command.

2. Install the required dependencies:

```bash
make requirements
```

This command installs the required Python dependencies listed in the `requirements.txt` file using `pip`. It ensures your environment has all the necessary libraries for the project to run smoothly.

3. (Optional) Set up a new Conda environment:

```bash
make create_environment
```

This command sets up a new Conda environment specifically for the project. It installs the required version of Python (3.10 in this case). You can activate this environment using `conda activate cataract-ai` after creation.

## 2. Dataset Information

The dataset consists of images of eyes labeled as either "normal" or "cataract." The images are divided into two main categories:

- **normal**: Images representing normal eyes.
- **cataract**: Images representing eyes with cataracts.
- **predict**: Where you will put unknown images for the trained model to predict

Additionally, there is a `predict` folder containing images for which predictions need to be made.

## 3. Download and Prepare the Dataset

1. Download the dataset ( > 500MB) from [Google Drive](https://drive.google.com/file/d/1QX6_PH7nBxPRkY-HtqNMs0TlVyybs3K3/view?usp=sharing).
2. Extract the dataset into the `data/raw` directory:

```bash
data/raw/
├── cataract/
│ ├── .gitignore
│ ├── image_x.jpg
│ ├── image_y.jpg
├── normal/
│ ├── .gitignore
│ ├── image_z.jpg
│ ├── image_t.jpg
├── predict/
│ ├── .gitignore
│ ├── image_j.jpg
│ ├── image_k.jpg
```

This step involves downloading the dataset from Google Drive and extracting it to the `data/raw/` directory. Make sure you preserve the folder structure, which includes `cataract`, `normal`, and `predict` subdirectories.

## 4. Training the Model

After setting up the dataset, you can train the model using the following commands:

### 4.1. Preprocess the data

```bash
make data
```

This command runs the `src/dataset.py` script, which preprocesses the raw image data. It might involve steps like resizing the images, converting them to grayscale, or performing other necessary transformations.

### 4.2. Extract features from the images

```bash
make features
```

This command runs the `src/features.py` script, which uses OpenCV (or other methods) to extract relevant features from the preprocessed images. These features are used for training the model and may include edge detection, texture features, and other image characteristics.

### 4.3. Train the model

```bash
make train
```

This command triggers the model training process using the `src/modeling/train.py` script. The model will be trained using a machine learning algorithm (Logistic Regression in this case). The trained model is then saved in the `models/` directory for future use.

## 5. Testing and Inference

Once the model is trained, you can run inference on new data using the following command:

### 5.1. Run predictions on new images

```bash
make predict
```

This command runs the `src/modeling/predict.py` script, which uses the trained model to make predictions on images in the `data/raw/predict/` folder. After running the prediction, the results are saved in a `predictions.csv` file in the `processed/` directory.

**Note:** The `data/raw/predict` folder is where you put images that you want to predict. By default, there are some sample images in that folder, but you can add your own images for prediction.

## 6. Exploratory Data Analysis (EDA)

1. `notebook/train.ipynb`: This Jupyter notebook is used to explore the dataset further. It performs exploratory data analysis (EDA) by generating graphs and visualizations. It also retrains the model while exploring the data. This notebook can only be run after preprocessing the data and extracting features.

2. `notebook/predict.ipynb`: This notebook performs the same task as the `predict.py` script, but it presents the results with more graphs and a better user interface. You can use this notebook to visualize model performance and make predictions on new images interactively.

## Additional Information

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
