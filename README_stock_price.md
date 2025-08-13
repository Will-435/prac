Prac
# Stock Prediction Using Decision Tree Regression


## Overview

This project uses a Decision Tree Regressor to predict the closing price of stocks based on historical stock data and sector information. The script reads an extended stock prices dataset, cleans the data, processes dates and categorical columns, and trains a regression model to predict closing prices.


## Features

### Data Cleaning

- Removes duplicate rows and duplicate columns.
- 
- Handles missing values by either filling with median or dropping rows, depending on the percentage of NaNs.


### Date Processing

- Converts date columns from strings to ordinal timestamp integers for model compatibility.


### Categorical Encoding

- Encodes string categorical columns into integer values to be used by the model.


### Model Training

- Tests multiple max_leaf_nodes values to find the optimal tree size.  
- Uses DecisionTreeRegressor from scikit-learn.  
- Evaluates models using Mean Absolute Error (MAE).


## Output

- Prints predicted closing prices.  
- Prints actual closing prices.  
- Displays MAE between predicted and actual values.


## Dependencies

Make sure you have the following Python libraries installed:

```bash
pip install pandas scikit-learn

## Dataset

The script expects a CSV file at:

`csv_files/stock_prices_extended.csv`

Required columns include (but are not limited to):

- date  
- open_price  
- high_price  
- low_price  
- trading_volume  
- volatility  
- volatility_dup  
- sector  
- close_price  

## How to Run

1.) Place the dataset in the correct path:

`csv_files/stock_prices_extended.csv`

2.) Run the script:

```bash
python stock_price_prediction.py

3.) The script will:

- Clean the data by removing duplicates and handling NaNs.
- Convert date strings to timestamps.
- Encode categorical columns.
- Train several decision tree models with different leaf node sizes.
- Select the best-performing model based on MAE.
- Output predicted and actual closing prices along with the MAE.

## Example Output

The predicted closing pricees are [ ... ]

The actual closing prices were [ ... ]

The mean absolute error between the two in 12.34

## Notes

- Duplicate columns and rows are removed to improve processing efficiency.
- NaN values are either filled with median or dropped depending on their proportion to preserve data quality.
- Dates must be converted to numeric ordinal values to be used as features.
- String categorical data is encoded to numeric values for compatibility with the regression model.
- Random state is fixed to 1 to ensure reproducibility.
- The model works best with numeric features only; categorical strings must be encoded prior to training.
