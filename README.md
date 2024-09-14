# Mortgage Prediction Web Application

This web application predicts the prepayment risk of mortgage loans. It allows users to input various mortgage-related parameters and generates both classification and regression predictions for the prepayment risk.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Overview

This project is a Flask-based web application that takes mortgage loan data as input and predicts:
- **Prepayment Risk Category** (Classification)
- **Prepayment Risk Percentage** (Regression)

The user can input values such as `Credit Score`, `LTV`, `DTI`, `Months in Repayment`, etc., and the model will return predictions based on the trained machine learning models.

## Features

- Input form for mortgage data such as `Credit Score`, `LTV`, `Months in Repayment`, etc.
- Predicts:
  - **Classification**: Risk category (e.g., Low, Medium, High)
  - **Regression**: Percentage risk of prepayment
- Provides feedback on time taken for prediction
- Simple and intuitive user interface

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS (custom)
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib (optional, if visualization is implemented)
- **Deployment**: Flask

## Project Structure

```bash
Mortgage-Prediction/
│
├── static/
│   └── css/
│       └── style.css     # Custom CSS for styling
│
├── templates/
│   └── index.html         # Main input form page
│   └── results.html      # Prediction results page
│
├── app.py                # Main Flask application file
├── model.py              # File containing model loading and prediction logic
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
