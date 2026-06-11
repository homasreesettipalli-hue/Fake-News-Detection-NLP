

## Overview

Fake News Detection is a Machine Learning and Natural Language Processing (NLP) project that classifies news articles as **REAL** or **FAKE**. The model is trained on a real-world news dataset using TF-IDF vectorization and Logistic Regression.

## Features

* News classification using Machine Learning
* Text preprocessing and cleaning
* TF-IDF feature extraction
* Real-time prediction from user input
* High accuracy on a real-world dataset

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLP
* TF-IDF Vectorization
* Logistic Regression

## Project Structure

```text
FakeNewsDetection
│
├── dataset
│   ├── Fake.csv
│   └── True.csv
│
├── model
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── train.py
├── predict.py
├── app.py
└── requirements.txt
```

## Dataset

The project uses the Fake and Real News Dataset containing thousands of news articles labeled as fake or real.

## Model Performance

* Algorithm: Logistic Regression
* Feature Extraction: TF-IDF
* Accuracy: 98.9%

## Installation

### Clone Repository

```bash
git clone https://github.com/homasreesettipalli-hue/Fake-News-Detection-NLP.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train.py
```

### Run Prediction

```bash
python predict.py
```

## Sample Usage

Input:

```text
Prime Minister Narendra Modi inaugurated a new infrastructure project in New Delhi.
```

Output:

```text
REAL NEWS
```

## Future Enhancements

* Flask Web Application
* Deep Learning Models
* News Website Integration
* Live News Verification System

## Author

Homasree Settipalli

GitHub: https://github.com/homasreesettipalli-hue
# Fake-News-Detection-NLP
