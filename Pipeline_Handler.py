#import libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
#Machine learning algorithms
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


# Preprocessing transformer
class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X):
        def prepayment(dti,income):
            if (dti<40):
              p =income/2
            else:
              p=income*3/4
            return p
        # Preprocessing steps
        X = X.copy()

        # Handling 'FirstTimeHomebuyer'
        X = X[X['FirstTimeHomebuyer'] != 'X']
        X['isHomeFirstBuyer'] = X['FirstTimeHomebuyer'].map({'Y': 1, 'N': 0})

        # Handling 'NumBorrowers'
        X = X[X['NumBorrowers'] != 'X ']
        X['NumBorrowers'] = X['NumBorrowers'].astype(int)

        # Channel frequency encoding
        for col in ['Channel']:
            freq_encoding = X[col].value_counts(normalize=True)
            X[f'{col}_Frequency'] = X[col].map(freq_encoding)



        # Encode categorical features
        X['LTV_Group_Encoded'] = X['LTV_range'].map({'Low LTV': 0, 'Medium LTV': 1, 'High LTV': 2})
        X['OCLTV_Group_Encoded'] = X['OCLTV_range'].map({'Low LTV': 0, 'Medium LTV': 1, 'High LTV': 2})
        X['FICO_Category_Encoded'] = X['Credit_range'].map({'Very Poor': 0, 'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4})
        X['YearsInRepayment_Group_Encoded'] = X['Repay_range'].map({'0-4 years': 0, '4-8 years': 1, '8-12 years': 2, '12-16 years': 3, '16-20 years': 4, '20+ years': 5})

        # DTI bins
        bins = [0, 20, 30, 40, 50, 100]
        labels = ['Low', 'Moderate', 'High', 'Very High', 'Extreme']
        X['DTIBin'] = pd.cut(X['DTI'], bins=bins, labels=labels, right=False)
        X['DTIBins'] = X['DTIBin'].map({'Low': 0, 'Moderate': 1, 'High': 2, 'Very High':3,'Extreme':4})

        # Calculate PaymentProgressRatio
        #X['PaymentProgressRatio'] = X['MonthsInRepayment'] / X['OrigLoanTerm']

        # Calculate MonthlyPayment
        p = X['OrigUPB']
        r = X['OrigInterestRate'] / (100 * 12)
        n = X['OrigLoanTerm']
        X['MonthlyPayment'] = round(p * r * ((1 + r) ** n / ((1 + r) ** n - 1)), 2)

        # Calculate CurrentPrincipal
        P = X['OrigUPB']
        r = X['OrigInterestRate'] / (100 * 12)
        E = X['MonthlyPayment']
        #M = X['MonthsInRepayment']
        #X['CurrentPrincipal'] = round((P * (1 + r) ** M) - (E * (((1 + r) ** M - 1) / r)), 2)

        # Calculate TotalAmount
        X['total_amount'] = X['MonthlyPayment'] * X['OrigLoanTerm']

        # Calculate InterestAmount
        X['interest_amount'] = X['total_amount'] - X['OrigUPB']

        # Calculate MonthlyIncome
        X['MonthlyIncome'] = round(X['MonthlyPayment'] / (X['DTI'] / 100), 2)

        X['prepayment']=np.vectorize(prepayment)(X['DTI'],X['MonthlyIncome'])
        X['prepayment']=(X['prepayment']*24)-(X['MonthlyPayment']*24)

        # Drop unneeded columns
        X = X.drop(columns=['MIP','Units', 'FirstTimeHomebuyer','LoanPurpose','SellerName','ServicerName',
                             'Channel','LTV_range','OCLTV_range', 'Credit_range',
                            'Repay_range','DTIBin'])

        # Handle any remaining missing values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.dropna(inplace=True)

        return X


class PipelineHandler:
    def __init__(self, clf, reg):
        # Initialize the two models
        self.clf = clf
        self.reg = reg
        # Preprocessing pipeline
        self.preprocessing = PreprocessingTransformer()
        self.standarscaler=StandardScaler()
        self.minmaxscaler=MinMaxScaler()
    def fit(self, X, y_class):
        # Preprocess the data
        X_processed = self.preprocessing.transform(X)
        y_class = y_class.loc[X_processed.index].values.ravel()  # Ensure 1D array

        # Select features for classification
        classification_features = ['MonthsDelinquent', 'FICO_Category_Encoded',
                            'YearsInRepayment_Group_Encoded', 'NumBorrowers',
                            'Channel_Frequency', 'LTV_Group_Encoded','OCLTV_Group_Encoded','isHomeFirstBuyer',
                            'OrigInterestRate']

        # Split the data for model training
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X_processed, y_class, X_processed['prepayment'], test_size=0.2, random_state=42)

        # Ensure that y_class_train and y_class_test are 1D arrays
        y_class_train = y_class_train.ravel()
        y_class_test = y_class_test.ravel()

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[classification_features] = self.standarscaler.fit_transform(X_train[classification_features])
        X_test_scaled[classification_features] = self.standarscaler.transform(X_test[classification_features])

        #********************************************************* Classification model
        # Fit classification model
        self.clf.fit(X_train_scaled[classification_features], y_class_train)  # Use scaled data

        # Predict delinquency status
        y_class_pred = self.clf.predict(X_test_scaled[classification_features])  # Use scaled data
        # For evaluation
        # Predict probabilities for ROC-AUC
        y_test_pred_prob = self.clf.predict_proba(X_test_scaled[classification_features])[:, 1]
        # Calculate scores
        testing_accuracy = accuracy_score(y_class_test, y_class_pred)
        roc_auc = roc_auc_score(y_class_test, y_test_pred_prob)
        f1 = f1_score(y_class_test, y_class_pred)
        print("Classification model results:")
        print("testing accuracy:", testing_accuracy)
        print("roc_auc:", roc_auc)
        print("f1_score:", f1)

        #******************************************************* Regression model

        # Select features for regression
        regression_features = ['OrigUPB', 'MonthlyPayment', 'total_amount',
                           'interest_amount', 'MonthlyIncome', 'DTIBins']
        X_train_scaled[regression_features] = self.minmaxscaler.fit_transform(X_train[regression_features])
        X_test_scaled[regression_features] = self.minmaxscaler.transform(X_test[regression_features])

        # Filter delinquent borrowers
        X_filtred_reg_train = X_train_scaled[y_class_train == 0]
        y_reg_filtered_train = y_reg_train[y_class_train == 0]  # Use y_class_train to filter y_reg_train

        X_filtred_reg_test = X_test_scaled[y_class_test == 0]
        y_reg_filtered_test = y_reg_test[y_class_test == 0]    # Use y_class_test to filter y_reg_test

        self.reg.fit(X_filtred_reg_train[regression_features], y_reg_filtered_train)
        # Predict Prepayment Amount
        y_reg_pred = self.reg.predict(X_filtred_reg_test[regression_features])
        # Calculate scores
        # Use mean squared error for regression problems
        b = mean_squared_error(y_reg_filtered_test, y_reg_pred, squared=False)   # RMSE for testing
        c = r2_score(y_reg_filtered_test, y_reg_pred)
        print("Regression model results:")
        print("RMSE", b)
        print("R2", c)

    def predict(self, X):
        # Preprocess the data
        print('X',X)
        X_processed = self.preprocessing.transform(X)
        print('X_processed',X_processed)
        # Select features for classification
        classification_features = ['MonthsDelinquent', 'FICO_Category_Encoded',
                            'YearsInRepayment_Group_Encoded', 'NumBorrowers',
                            'Channel_Frequency', 'LTV_Group_Encoded','OCLTV_Group_Encoded','isHomeFirstBuyer',
                            'OrigInterestRate']
        # Select features for regression model
        regression_features = ['OrigUPB', 'MonthlyPayment', 'total_amount',
                           'interest_amount', 'MonthlyIncome', 'DTIBins']
        # We need to standardize the data
        X_processed[classification_features] = self.standarscaler.transform(X_processed[classification_features])
        # We need to standardize the data
        X_processed[regression_features] = self.minmaxscaler.transform(X_processed[regression_features])

        #*********************Classification model prediction****************************
        # Predict delinquency status
        y_class_pred = self.clf.predict(X_processed[classification_features])

        #***********************Regression model classification**************************
        # Filter data for non-delinquent predictions
        X_filtered_reg_new = X_processed[y_class_pred == 0]

        if X_filtered_reg_new.shape[0] > 0:
        # Predict repayment amount for non-delinquent borrowers
           y_reg_pred_new = self.reg.predict(X_filtered_reg_new[regression_features])
        else:
           y_reg_pred_new = np.array([])  # No non-delinquent borrowers in the new data


        return y_class_pred, y_reg_pred_new

