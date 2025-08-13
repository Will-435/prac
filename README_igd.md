# prac
Just practicing

Loan Loss Prediction Using Decision Tree Regression

**Overview**

This project uses a Decision Tree Regressor to predict the Loss Given Default (LGD) for a borrower based on loan and borrower attributes.
The script reads a dataset of loans, cleans the data, selects features, and trains a regression model to predict LGD.

**Features**

*Data Cleaning*
* Removes duplicate rows.
* Removes the region column (categorical).
* Fills missing numeric values with the median of their respective columns.

*Model Training*
* Uses DecisionTreeRegressor from scikit-learn.
* Tests multiple max_leaf_nodes values to find the optimal setting.
* Evaluates models using Mean Absolute Error (MAE).

*Output*
* Prints predicted LGD values.
* Prints actual LGD values from the dataset.
* Displays MAE between predictions and actual data.

**Dependencies** 

Make sure you have the following Python libraries installed:

pip install pandas scikit-learn

**Dataset**

The script expects a CSV file at:

csv_files/loan_loss.csv

*Required columns:*

* annual_income
* loan_amount
* interest_rate
* credit_score
* employment_years
* loan_amount_dup
* default_loss

**How to Run**

1.) Place the dataset in the correct path:

csv_files/loan_loss.csv

2.) Run the script:

python loan_loss_prediction.py

3.) The script will:

* Clean the data.
* Train several decision tree models with different leaf sizes.
* Select the best-performing model.
* Output:
    * Predicted LGD values.
    * Actual LGD values.
    * Mean Absolute Error (MAE).
 

*Example Output*

The predicted loss given default: [ ... ]

The correct loss given default is: [ ... ]

Mean Absolute Error between prediction and collected data is: 1234.56

**Notes**

* The column loan_amount_dup appears to be a duplicate of loan_amount — this is intentional for demonstration but could be removed for better performance.
* random_state=1 ensures reproducible results.
* The model only works with numeric features — categorical variables (like region) must be encoded before use.
