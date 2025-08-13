Prac
# Stock Prediction Using Decision Tree Regression

## Overview

This project uses a Decision Tree Regressor to predict the closing price of stocks based on historical stock data and sector information. The script reads an extended stock prices dataset, cleans the data, processes dates and categorical columns, and trains a regression model to predict closing prices.

## Features

### Data Cleaning

- Removes duplicate rows and duplicate columns.  
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
