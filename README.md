# 🦋 Butterfly Species Classification

## Project Overview

This project is a deep learning-based web application that classifies butterfly species from uploaded images. The model was trained using the EfficientNetB0 architecture and deployed with Flask, allowing users to upload a butterfly image and receive a predicted species.



## Objective

The objective of this project is to build an image classification system capable of identifying butterfly species using Convolutional Neural Networks (CNNs) and transfer learning.



## Dataset

- Butterfly Image Classification Dataset
- 75 butterfly species
- Images divided into training, validation, and testing sets



## Technologies Used

- Python
- TensorFlow / Keras
- EfficientNetB0
- NumPy
- Pillow
- OpenCV
- Flask
- HTML



## Project Structure

```
Butterfly_Flask_Project/
│
├── app.py
├── class_indices.json
├── butterfly_model.keras
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
```



## Model

- Transfer Learning using EfficientNetB0
- Image Size: 224 × 224
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Evaluation Metrics: Accuracy



## Features

- Upload butterfly images
- Predict butterfly species
- Modern Flask web interface
- Image preview before prediction


## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the Flask server

```bash
python app.py
```

### Open your browser

```
http://127.0.0.1:5000
```

---

## Example Prediction

Input Image:

- Monarch Butterfly

Prediction:

```
Predicted: MONARCH (0.98)
```



## Author

Memory Akter
