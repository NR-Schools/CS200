# Corn Leaf Disease Classifier

## Overview

The Corn Leaf Disease Classifier is a web application designed to classify corn leaf diseases using the VGNet Model. The application allows users to upload an image of a corn leaf and receive a classification result, indicating the type of disease and the model's confidence in its prediction.

## Features

- Upload an image of a corn leaf.
- Preview the image before uploading.
- Classify the image using the VGNet Model.
- Display classification results.

## Technologies Used

- **Flask**: Python web framework used for building the application.
- **Tensorflow and Keras**: Deep learning library used for the VGNet Model.
- **HTML/CSS/JavaScript**: For the frontend and user interface.

## Setup

### Prerequisites

- Python 3.x
- Flask
- Keras (with a trained VGNet Model)
- Pillow (for image handling)

### Installation

1. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment:**

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

4. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

5. **Add your trained VGNet Model(s)**:
    - Place your models in the `app/models` directory.

### Running the Application

1. **Run the Flask application:**

    ```bash
    python -m flask --app app run --port 8000
    ```

2. **Open your web browser and navigate to:**

    ```
    http://127.0.0.1:8000
    ```

## How to Use

1. Go to the home page.
2. Choose an image file of a corn leaf to upload.
3. Click the "Upload" button.
4. The application will display the image preview and classify the image using the VGNet Model.
5. The classification result, including the class and confidence, will be shown.

## Folder Structure

- `app/`
  - `__init__.py`: Initializes the Flask application and routes.
  - `model.py`: Contains the `classify_image` function and model loading logic.
- `templates/`
  - `index.html`: Upload page with image preview.
  - `uploaded.html`: Displays the classification result and includes a back button.
- `static/`
  - `css/styles.css`: Styles for the application.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file.

## Model Description

The VGNet Model is a deep learning model designed for image classification tasks. It uses advanced convolutional neural network techniques to achieve high accuracy in identifying and classifying images. This application uses the VGNet Model to detect and classify diseases in corn leaves.
