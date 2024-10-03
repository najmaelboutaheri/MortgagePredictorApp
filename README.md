# Mortgage Prepayment Prediction App

This is a Flask-based web application for predicting mortgage loan classifications and regressions, specifically prepayment and delinquency risks. The application uses a pre-trained machine learning model and takes various input features related to the mortgage, borrower, and loan properties to perform predictions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Form Inputs](#form-inputs)
- [Logging](#logging)
- [Model Overview](#model-overview)
- [Contributing](#contributing)
- [License](#license)

## Overview
The application accepts inputs related to mortgage loans, such as borrower attributes, loan details, and mortgage history, to predict the prepayment and delinquency likelihood using machine learning models.

## Features
- Web-based form to input mortgage loan details.
- Backend Flask application that handles form data and performs predictions.
- Categorical and numerical validation for user input.
- Logging mechanism to track user interaction and any errors.
- Pre-trained machine learning model using scikit-learn for prediction.

## Setup and Installation

### Prerequisites
Make sure you have the following installed on your machine:
- Python 3.x
- Flask
- scikit-learn
- pandas
- joblib
- HTML templates

### Clone the Repository
```bash
git clone https://github.com/yourusername/mortgage-prepayment-prediction.git
cd mortgage-prepayment-prediction
```

### Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Add the Pre-trained Model
Download the pre-trained model file **mortgage_pipeline_model_.pkl** and place it in the **models/** directory.

### Run the Application
Start the Flask application by running:
```bash
python app.py
```
Open your browser and go to ```http://localhost:5000``` to access the web interface.

### Usage
1. Open the web interface.
2. Fill in the required fields with the mortgage loan information.
3. Click "Submit" to make a prediction.
3. The application will return the prediction results for both classification (e.g., prepayment or delinquency risk) and regression (e.g., estimated future payments).
If the input values are invalid or out of range, the app will notify you of the errors.

### Form Inputs
The form expects the following input fields:

#### Categorical Features:
- Channel: Retail, Broker, Correspondent
- FirstTimeHomebuyer: Y, N
- LoanPurpose: P, R
- SellerName: AC, NO, FL, CO
- ServicerName: GMACMTGECORP, WELLSFARGOBANKNA, etc.
- Credit_range: Very Poor, Poor, Fair, Good, Excellent
- LTV_range: Low LTV, Medium LTV, High LTV
- OCLTV_range: Low LTV, Medium LTV, High LTV
- Repay_range: 0-4 years, 4-8 years, etc.
#### Numerical Features:
- MonthsDelinquent: Numeric value between 0 and 120
- DTI: Numeric value between 0 and 100
- OrigInterestRate: Numeric value between 0 and 20
- OrigUPB: Numeric value up to 1,000,000
- OrigLoanTerm: Numeric value between 0 and 360
- Units: Numeric value between 1 and 10
- MIP: Numeric value between 0 and 1
- NumBorrowers: Numeric value between 1 and 10
#### Logging
The application uses Python's built-in logging library for debugging and error tracking. Logs are output to the console by default.

### Model Overview
The machine learning model used in this app is built using scikit-learn and consists of both classification and regression predictions. The model has been trained to predict:

- Classification: Prepayment or delinquency risk of the loan.
- Regression: Estimated future mortgage-related metrics like payments and prepayment.
#### Preprocessing Steps
The preprocessing steps applied to the data include:

1. Frequency encoding for categorical variables.
2. Binning of numerical features (e.g., DTI bins).
3. Creation of new features such as MonthlyPayment, InterestAmount, and Prepayment.
4. The model pipeline combines preprocessing with machine learning algorithms such as Logistic Regression for classification and Lasso Regression for regression tasks.

### Contributing
If you'd like to contribute to this project:

1. Fork the repository.
2. Create a new feature branch (git checkout -b feature-branch).
3. Commit your changes (git commit -m "Add feature").
4. Push to the branch (git push origin feature-branch).
5. Open a Pull Request.

### License

This project is licensed under the MIT License.
License content:
```
MIT License

Copyright (c) 2024 NAJMA EL BOUTAHERI.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
### Contact:
- **[Email](najma.elboutaheri@etu.uae.ac.ma)** 
- **[Linkdin profile](https://www.linkedin.com/in/najma-el-boutaheri-8185a1267/)** 
