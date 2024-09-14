from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import time
import logging
from Pipeline_Handler import PipelineHandler, PreprocessingTransformer
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the pre-trained model from the pickle file
logger.debug("Loading model...")
with open('models/mortgage_pipeline_model_.pkl', 'rb') as f:
    model = joblib.load(f)
logger.debug("Model loaded successfully.")

# Define categorical and numerical columns for form input
cat_cols = {
    'Channel': ['Retail', 'Broker', 'Correspondent'],
    'FirstTimeHomebuyer': ['Y', 'N'],
    'LoanPurpose': ['P', 'R'],  # Add relevant options
    'SellerName': ['AC', 'NO', 'FL', 'CO'],  # Add relevant options
    'ServicerName': ['GMACMTGECORP', 'WELLSFARGOBANKNA', 'CHASEMANHATTANMTGECO', 'COUNTRYWIDE']  # Add relevant options
    ,'Credit_range':['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'],
    'LTV_range':['Low LTV', 'Medium LTV', 'High LTV'],
    'OCLTV_range':['Low LTV', 'Medium LTV', 'High LTV'],
    'Repay_range':['0-4 years', '4-8 years', '8-12 years', '12-16 years', '16-20 years', '20+ years']
}

num_cols = {
    'MonthsDelinquent': 0,
    'DTI': 36,
    'OrigInterestRate': 3.5,
    'OrigUPB': 300000,
    'OrigLoanTerm': 360,
    'Units': 1,  # Add default value for units
    'MIP': 0,  # Add default value for MIP
     'NumBorrowers':1,  
     
}

@app.route('/')
def home():
    logger.debug("Home route accessed.")
    return render_template('index.html', cat_cols=cat_cols, num_cols=num_cols)

@app.route('/predict', methods=['POST'])
def predict():
    try:
       logger.debug("Prediction route accessed.")
       start_time = time.time()

       # Collect form data
       form_data = {
            "MonthsDelinquent": [int(request.form['MonthsDelinquent'])],
            "DTI": [int(request.form['DTI'])],
            "OrigInterestRate": [float(request.form['OrigInterestRate'])],
            "OrigUPB": [int(request.form['OrigUPB'])],
            "OrigLoanTerm": [int(request.form['OrigLoanTerm'])],
            "Units": [int(request.form['Units'])],
            "MIP": [float(request.form['MIP'])],
            "Channel": [request.form['Channel']],
            "FirstTimeHomebuyer": [request.form['FirstTimeHomebuyer']],
            "LoanPurpose": [request.form['LoanPurpose']],
            "SellerName": [request.form['SellerName']],
            "ServicerName": [request.form['ServicerName']],
            "Credit_range": [request.form['Credit_range']],  # Update as necessary
            "LTV_range": [request.form['LTV_range']],  # Update as necessary
            "OCLTV_range": [request.form['OCLTV_range']],  # Update as necessary
            "NumBorrowers": [float(request.form['NumBorrowers'])],
            "Repay_range": [request.form['Repay_range']] # Update as necessary
           # "EverDelinquent": [int(request.form['EverDelinquent'])]
            }

       logger.debug(f"Form data received: {form_data}")

        # Input validation
       errors = []

        # Validate numerical inputs
       if not (0 <= form_data['MonthsDelinquent'][0] <= 120):
            errors.append("MonthsDelinquent must be between 0 and 120.")
       if not (0 <= form_data['DTI'][0] <= 100):
            errors.append("DTI must be between 0 and 100.")
       if not (0 <= form_data['OrigInterestRate'][0] <= 20):
            errors.append("OrigInterestRate must be between 0 and 20.")
       if not (0 <= form_data['OrigUPB'][0] <= 1_000_000):
            errors.append("OrigUPB must be between 0 and 1,000,000.")
       if not (0 <= form_data['OrigLoanTerm'][0] <= 360):
            errors.append("OrigLoanTerm must be between 0 and 360.")
       if not (1 <= form_data['Units'][0] <= 10):
            errors.append("Units must be between 1 and 10.")
       if not (0 <= form_data['MIP'][0] <= 1):  # Example validation
            errors.append("MIP must be between 0 and 1.")
       if not (0 <= form_data['NumBorrowers'][0] <= 10):  # Example validation
            errors.append("NumBorrowers must be between 0 and 10.")
       

       if errors:
            error_message = "Validation errors: " + ", ".join(errors)
            logger.error(error_message)
            return render_template('index.html', error_message=error_message,cat_cols=cat_cols, num_cols=form_data)

       # Convert form data to DataFrame
       df = pd.DataFrame(form_data)

       # Perform prediction
       try:
            logger.debug("Making predictions.")
            y_class_pred, y_reg_pred_new = model.predict(df[["FirstTimeHomebuyer","MIP","Units","OrigUPB",
                                                             "OrigInterestRate", "OrigLoanTerm", "Channel", "LoanPurpose", "NumBorrowers", "SellerName",
                                                               "ServicerName", "Credit_range",
                                                             "LTV_range", "OCLTV_range","DTI","Repay_range","MonthsDelinquent"]])
       except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return render_template('index.html', error_message=f"An error occurred: {str(e)}",cat_cols=cat_cols, num_cols=form_data)
      
       end_time = time.time()
       elapsed_time = end_time - start_time
       logger.debug(f"Prediction completed in {elapsed_time:.2f} seconds.")
       if len(y_reg_pred_new) > 0:
           return render_template('result.html', 
                           classification_prediction=y_class_pred[0], 
                           regression_prediction=y_reg_pred_new[0],
                           time_taken=elapsed_time)
       else:
           error_message = "Delinquent borrowers have no prepayment"
           return render_template('result.html', 
                           classification_prediction=y_class_pred[0],
                           time_taken=elapsed_time,error_message=error_message)
    except Exception as e:
           # Return prediction result to the HTML page
           logger.error(f"Error in the prediction process: {e}")
           return render_template('index.html', error_message=f"An error occurred: {str(e)}",cat_cols=cat_cols, num_cols=form_data)

if __name__ == '__main__':
    logger.debug("Starting the Flask app.")
    app.run(host='0.0.0.0', port=5000, debug=True)
