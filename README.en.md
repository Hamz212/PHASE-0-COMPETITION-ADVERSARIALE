# Phase 0 – Intrusion Detection System (IDS) Competition

## Overview

Welcome to Phase 0 of the Adversarial Machine Learning Competition!

In this phase, your goal is to develop a **Feed-Forward Neural Network (FFNN)** capable of classifying network traffic as either:

- **0** – Benign
- **1** – Malicious

You will train your model using the provided training dataset and submit your trained model for evaluation against a hidden test set.

---

# Learning Objectives

By completing this phase you will gain experience with:

- Loading and preprocessing tabular datasets
- Training neural networks using PyTorch
- Hyperparameter tuning
- Model evaluation
- Building an Intrusion Detection System (IDS)

---

# Provided Files

```
starter_submission/
├── scaledXTrain.csv
├── yTrain.csv
├── model.py
├── validateSubmission.py
├── requirements.txt
└── README.md
```

---

# Competition Rules

You **may** modify:

- Neural network architecture
- Number of hidden layers
- Number of neurons
- Activation functions
- Optimizer
- Learning rate
- Batch size
- Number of epochs
- Early stopping
- Regularization
- Dropout
- Classification threshold
- Any other FFNN training techniques

You **may not** use:

- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformers
- Decision Trees
- Random Forests
- Gradient Boosted Trees
- Support Vector Machines
- Any architecture that is **not** a Feed-Forward Neural Network (MLP)

---

# Training

Implement your training procedure inside:

```
model.py
```

The provided starter code contains:

- IDSModel class
- forward()
- fit()
- predict()

Modify these functions however you wish while remaining within the competition rules.

---

# Validation

Before submitting, validate your submission by running:

```
python validateSubmission.py
```

The validation script checks that:

- submission.zip exists
- model.py exists
- model.pth exists
- IDSModel can be imported
- Model weights load successfully
- predict() executes correctly
- Predictions have the correct shape
- Predictions are binary (0 or 1)

A successful validation does **not** guarantee a high score — it only verifies that your submission can be evaluated.

---

# Submission Format

Your final submission **must** be named:

```
submission.zip
```

and contain exactly:

```
submission.zip
├── model.py
└── model.pth
```

No additional files are required.

---

# Evaluation

Your submission will be evaluated against a **hidden test dataset** that is **not** released to students.

The evaluation process is:

```
submission.zip
        │
        ▼
Extract submission
        │
        ▼
Import model.py
        │
        ▼
Load model.pth
        │
        ▼
Predict hidden test set
        │
        ▼
Compute evaluation metrics
```

The organizer's evaluation script follows this workflow to load your model and compute the final scores. 

---

# Scoring

The following metrics will be reported:

- Accuracy
- Precision
- Recall
- F1 Score

The primary competition metric is:

> **F1 Score**

---

# Tips

- Start with the provided model before experimenting.
- Change one thing at a time.
- Monitor performance during training.
- Keep your implementation simple before trying more advanced ideas.
- Always validate your submission before submitting.

Good luck, and have fun!
