# SMS Spam Classification

A machine learning project that classifies SMS messages as spam or ham using TF-IDF and Logistic Regression.

## Overview

The goal is to detect unwanted messages while minimizing false positives.

The project explores text preprocessing, feature extraction, classification algorithms, and model evaluation.

## Tech Stack

- Python
- Pandas / NumPy
- Matplotlib
- Scikit-learn
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes

## Project Structure

```text
sms-spam-classifier/
├── data/
├── notebooks/
│   └── spam_classification.ipynb
├── models/
├── README.md
├── requirements.txt
└── .gitignore
```

## Machine Learning Workflow

1. Data cleaning and exploratory data analysis
2. Text vectorization using TF-IDF
3. Model training with Naive Bayes and Logistic Regression
4. Model evaluation using Precision, Recall, and Confusion Matrix
5. Decision threshold analysis
6. Word and character n-gram experiments
7. Scikit-learn Pipeline implementation

## Results

| Model                                 | Precision (Spam) | Recall (Spam) |
| ------------------------------------- | ---------------: | ------------: |
| Multinomial Naive Bayes               |           100.0% |         61.8% |
| Logistic Regression                   |            99.0% |         74.8% |
| Logistic Regression (Threshold = 0.3) |            94.1% |         84.7% |

The results demonstrate the trade-off between precision and recall.

**Note:** Results are exploratory because the test set was reused during model and threshold experiments.

## Dataset

SMS Spam Collection Dataset, available on [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

## Future Improvements

- Dedicated validation set
- Hyperparameter tuning
- Model serialization with Joblib
- REST API with FastAPI
- Evaluation on new, unseen messages

## Author

Emre Ertuğrul
