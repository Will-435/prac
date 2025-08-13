import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


"""
This function will intake a data frame and remove all rows and columns 
that are duplicates within the data frame. 
Cleaning the data this way improves processing time by cutting out unnecessary work.
"""
def drop_duplicate_rows_cols(df):

    # remove duplicated rows
    df_1 = df.drop_duplicates()

    # remove duplicated columns
    df_2_transpose = df_1.T.drop_duplicates()
    df_2 = df_2_transpose.T

    return df_2


"""
This function inputs a dataframe with not a number (NaN) elements and handles them accordingly.
If NaN elements are below 20%, we replace them with the column mean in order to make the most of our 
data without skewing the results. If the NaN elements are over 20%, their rows are removed.
This is necessary to stop errors later on when performing calculations through built-in functions.
"""
def nan_handling(df):

    columns = df.columns
    for col in columns:

        boolean_values = df[col].isna() # creating a boolean matrix, where True represents NaN element
        NaN_sum = boolean_values.sum()
        num_elements = df[col].size

        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df = df.dropna(subset=[col]) 

        elif pd.api.types.is_any_real_numeric_dtype(df[col]) and NaN_sum > num_elements * 0.2:
            df = df.dropna(subset=[col]) # remove all rows with NaN elements
    
        else:
            df[col] = df[col].fillna(df[col].median) # fills that column with its respective median from the df.median() series

    return df


"""
This function takes a dataframe with dates as strings and converts them to timestamp objects.
Iterating over each column is fast. Only if the element fits the condition, a date will be converted.
This is necessary for the date column to be interpreted in our model as numeric, which is essential.
""" 
def convert_dates_to_timestamp(df):

    columns = df.columns
    for col in columns:
        # if the elements of the column are possibly a string 
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            try:
                counted_days = pd.to_datetime(df[col]).map(pd.Timestamp.toordinal)
                df[col] = counted_days

            except Exception:
                pass

    return df


"""
This function takes a data frame with columns of strings and encodes them as numbers.
The dictionary assigns each unique string an integer.
This is necessary so that our model can associate certain sectors with patterns in our data.
We don't need to consider dates; they have already been turned into integers.
"""
def encode_strings_as_integers(df):

    columns = df.columns
    for col in columns:

        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            unique_strings = df[col].unique()

            keys = {string: index for index, string in enumerate(unique_strings)}

            encoded = df[col].map(keys.get)
            df[col] = encoded

    return df


"""
This function tries different values of max leaf nodes to find the optimum using mean error.
Creating its model ensures we don't test one model once, resulting in overfitting.
This is important for building confidence in our final model.
"""
def calculate_min_mae(max_leaf_nodes, test_x, test_y):

    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 1)
    model.fit(test_x, test_y)

    prediction_y = model.predict(test_x)
    mae = mean_absolute_error(test_y, prediction_y)

    return mae


df_0 = pd.read_csv('csv_files/stock_prices_extended.csv')

df_1 = drop_duplicate_rows_cols(df_0)

df_2 = nan_handling(df_1)

df_3 = convert_dates_to_timestamp(df_2)

df_4 = encode_strings_as_integers(df_3)

print(df_4.columns)

# assign the features and targets
features = ['date', 'open_price', 'high_price', 'low_price', 'trading_volume', 'volatility', 'volatility_dup', 'sector']
x = df_4[features]
y = df_4['close_price']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 1)

# sample leaf values to trial, avoiding over or underfitting in the final model
leaf_values = [5, 10, 50, 100, 500, 1000, 5000]

# we need a way to extract the leaf value from the min mae value so dictionary
keys = {value: calculate_min_mae(value, test_x, test_y) for value in leaf_values}
optimum_leaf_value = min(keys, key = keys.get)

final_model = DecisionTreeRegressor(max_leaf_nodes = optimum_leaf_value, random_state = 1)
final_model.fit(train_x, train_y)

prediction_y = final_model.predict(test_x)

msg_1 = f'The predicted closing pricees are {prediction_y}'
msg_2 = f'The actual closing prices were {test_y}'

difference = abs(test_y - prediction_y)

mea_prediction = mean_absolute_error(test_y, prediction_y)
msg_3 = f'The mean absolute error between the two in {mea_prediction}'

print(msg_1, '\n')
print(msg_2, '\n')
print(msg_3, '\n')