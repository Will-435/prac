import pandas as pd
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

"""
Task 1 - Create a simple decision tree regressor model to predict the loss given default for a borrower
    * Remove duplicate rows
    * Remove duplicate loan_amount column
    * Drop rows with NaN values
    * Encode categorical region
Target (y): default_loss
"""


def calculate_mae(max_leaf_nodes, train_x, test_x, train_y, test_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state = 1)
    model.fit(train_x, train_y)

    predict_y = model.predict(test_x)
    mae = mean_absolute_error(test_y, predict_y)

    return mae

csv_file_path = 'csv_files/loan_loss.csv'
client_df = pd.read_csv(csv_file_path)

# drop duplicate rows
df1 = client_df.drop_duplicates()
# drop unnecessary columns
df2 = df1.drop('region', axis = 1)
# fill empty elements with mean of their culumn using a series
df3 = df2.fillna(df2.median(numeric_only = True))

# assign the features and the target fpor the model (x and y respectively)
features = ['annual_income', 'loan_amount', 'interest_rate', 'credit_score', 'employment_years', 'loan_amount_dup']
x = df3[features]
y = df3['default_loss']

# splitting our data insto training and testing sets
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 1)

# finding the ideal amount of leaf nodes to avoid underfitting or overfitting
leaf_amounts = [5, 10, 50, 100, 500, 1000, 5000]
dictionary = {amount: calculate_mae(amount, train_x, test_x, train_y, test_y) for amount in leaf_amounts}
ideal_leaf_num = min(dictionary, key = dictionary.get)

# creating our final model 
lgd_model = DecisionTreeRegressor(max_leaf_nodes = ideal_leaf_num, random_state = 1)
lgd_model.fit(train_x, train_y)
lgd_prediction = lgd_model.predict(test_x)

# printing the predicted lgd, the actual lgd, and the difference to see how close our model is
print(f'The predicted loss given default: {lgd_prediction} \n')
print(f'The correct loss given default is: {test_y} \n')
mae = mean_absolute_error(test_y, lgd_prediction)
print(f"Mean Absolute Error between prediction and collected data is: {mae}")
