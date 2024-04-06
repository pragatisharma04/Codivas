import json
import time
import datetime
import numpy as np
import pandas as pd
from flask import Flask, request
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

# Load model parameters from a CSV file
model_params = pd.read_csv('model_params.csv')

# n_estimators=model_params['i_forest']['n_estimators']: This sets the number of trees in the Isolation Forest model. The n_estimators parameter determines the number of decision trees that will be used in the ensemble. The value for this parameter is taken from the 'n_estimators' key in the 'i_forest' dictionary of the model_params dictionary.
# contamination=model_params['i_forest']['contamination']: This sets the expected proportion of outliers in the dataset. The contamination parameter specifies the proportion of outliers in the dataset, which is used to determine the threshold for identifying anomalies. The value for this parameter is taken from the 'contamination' key in the 'i_forest' dictionary of the model_params dictionary.
# random_state=model_params['i_forest']['random_state']: This sets the random state or seed for the Isolation Forest model. The random_state parameter ensures that the model's results are reproducible by setting a specific seed for the random number generator. The value for this parameter is taken from the 'random_state' key in the 'i_forest' dictionary of the model_params dictionary.
# i_forest = IsolationForest(...): This line creates an instance of the Isolation Forest model with the specified parameters and assigns it to the variable i_forest.

# Initialize the Isolation Forest model
i_forest = IsolationForest(n_estimators=model_params['i_forest']['n_estimators'],
                           contamination=model_params['i_forest']['contamination'],
                           random_state=model_params['i_forest']['random_state'])

# Initialize variables to store transaction data
transaction_data = []
transaction_amounts = []
transaction_locations = []

def is_coherent_pattern(transaction_amount, location, transaction_age_hours, card_balance, merchant_category_code, transaction_history):
    """
    Check if the transactions from a card follow a coherent pattern based on the last 12-hour, 1-day, and 7-day windows.

    Parameters:
    transaction_amount (float): The amount of the current transaction.
    location (tuple): The location of the current transaction.
    transaction_age_hours (float): The age of the current transaction in hours.
    card_balance (float): The current balance of the card.
    merchant_category_code (int): The merchant category code of the current transaction.
    transaction_history (DataFrame): The transaction history of the card.

    Returns:
    bool: True if the transactions follow a coherent pattern, False otherwise.
    """

    # Filter transaction history based on the last 12-hour, 1-day, and 7-day windows
    last_12_hours = transaction_history[(transaction_history['transaction_time'] >= transaction_age_hours - 12) & (transaction_history['transaction_time'] <= transaction_age_hours)]
    last_1_day = transaction_history[(transaction_history['transaction_time'] >= transaction_age_hours - 24) & (transaction_history['transaction_time'] <= transaction_age_hours)]
    last_7_days = transaction_history[(transaction_history['transaction_time'] >= transaction_age_hours - 168) & (transaction_history['transaction_time'] <= transaction_age_hours)]

    # Check if the transactions follow a coherent pattern based on the last 12-hour window
    if not is_coherent_pattern_window(transaction_amount, location, transaction_age_hours, card_balance, merchant_category_code, last_12_hours):
        return False

    # Check if the transactions follow a coherent pattern based on the last 1-day window
    if not is_coherent_pattern_window(transaction_amount, location, transaction_age_hours, card_balance, merchant_category_code, last_1_day):
        return False

    # Check if the transactions follow a coherent pattern based on the last 7-day window
    if not is_coherent_pattern_window(transaction_amount, location, transaction_age_hours, card_balance, merchant_category_code, last_7_days):
        return False

    # If the transactions follow a coherent pattern in all three windows, return True
    return True

def is_coherent_pattern_window(transaction_amount, location, transaction_age_hours, card_balance, merchant_category_code, transaction_window):
    """
    Check if the transactions from a card follow a coherent pattern based on a given window.

    Parameters:
    transaction_amount (float): The amount of the current transaction.
    location (tuple): The location of the current transaction.
    transaction_age_hours (float): The age of the current transaction in hours.
    card_balance (float): The current balance of the card.
    merchant_category_code (int): The merchant category code of the current transaction.
    transaction_window (DataFrame): The transaction history of the card within the given window.

    Returns:
    bool: True if the transactions follow a coherent pattern, False otherwise.
    """

    # Check if the transaction amount is within a reasonable range based on the card balance
    if transaction_amount >= 0.7 * card_balance and card_balance >= 300000:
        return False

    # Check if the transaction location is within a reasonable distance from the previous transaction location

    # and np.linalg.norm(np.array(location) - np.array(transaction_window['location'].iloc[-1])) >= 200000:
    # This part of the code calculates the Euclidean distance between the location variable and the location of the last transaction in the transaction_window DataFrame.
    # np.array(location) converts the location variable (which may be a list or a tuple) into a NumPy array.
    # np.array(transaction_window['location'].iloc[-1]) extracts the location of the last transaction in the transaction_window DataFrame and converts it into a NumPy array.
    # np.linalg.norm() calculates the Euclidean distance between the two NumPy arrays.
    # If the Euclidean distance is greater than or equal to 200,000, the condition will be True.

    # The syntax for using .iloc is df.iloc[row_start:row_end, column_start:column_end], where df is the DataFrame you want to select from.
    # [-1]: This part of the expression selects the last row of the DataFrame. The negative index -1 refers to the last element in the DataFrame, -2 refers to the second-to-last element, and so on.


    if len(transaction_window) > 0 and np.linalg.norm(np.array(location) - np.array(transaction_window['location'].iloc[-1])) >= 200000:
        return False

    # Check if the transaction merchant category code is consistent with the previous transaction merchant category code
    if len(transaction_window) > 0 and transaction_window['merchant_category_code'].iloc[-1] != merchant_category_code:
        return False

    # If the transaction follows a coherent pattern with the previous transactions in the window, return True
    return True

def is_coherent_pattern_with_merchant_category_code(merchant_category_code, transaction_age_hours, card_id, transaction_history):
    """
    Check if the transaction follows a coherent pattern with the merchant category code of the last 3-day, 7-day, and 30-day windows for the card.

    Parameters:
    merchant_category_code (int): The merchant category code of the current transaction.
    transaction_age_hours (float): The age of the current transaction in hours.
    card_id (str): The unique identifier of the card.
    transaction_history (DataFrame): The transaction history of the card.

    Returns:
    bool: True if the transaction follows a coherent pattern, False otherwise.
    """

    # Filter transaction history based on the last 3-day, 7-day, and 30-day windows
    last_3_days = transaction_history[(transaction_history['transaction_time'] >= transaction_age_hours - 72) & (transaction_history['transaction_time'] <= transaction_age_hours)]
    last_7_days = transaction_history[(transaction_history['transaction_time'] >= transaction_age_hours - 168) & (transaction_history['transaction_time'] <= transaction_age_hours)]
    last_30_days = transaction_history[(transaction_history['transaction_time'] >= transaction_age_hours - 720) & (transaction_history['transaction_time'] <= transaction_age_hours)]

    # Check if the transaction merchant category code is coherent with the last 3-day window
    if not is_coherent_merchant_category_code(merchant_category_code, card_id, last_3_days):
        return False

    # Check if the transaction merchant category code is coherent with the last 7-day window
    if not is_coherent_merchant_category_code(merchant_category_code, card_id, last_7_days):
        return False

    # Check if the transaction merchant category code is coherent with the last 30-day window
    if not is_coherent_merchant_category_code(merchant_category_code, card_id, last_30_days):
        return False

    # If the transaction follows a coherent pattern in all three windows, return True
    return True

def is_coherent_merchant_category_code(merchant_category_code, card_id, transaction_window):
    """
    Check if the transaction follows a coherent pattern with the merchant category code in a given window.

    Parameters:
    merchant_category_code (int): The merchant category code of the current transaction.
    card_id (str): The unique identifier of the card.
    transaction_window (DataFrame): The transaction history of the card within the given window.

    Returns:
    bool: True if the transaction follows a coherent pattern, False otherwise.
    """

    # Check if there are previous transactions in the window for the card
    if len(transaction_window) == 0:
        return True

    # Check if the merchant category code of the current transaction is consistent with the previous transactions

    # transaction_window[transaction_window['card_id'] == card_id]:
    # This part of the code filters the transaction_window DataFrame to only include rows where the 'card_id' column matches the card_id variable.
    # The result is a new DataFrame that contains only the relevant transactions for the given card_id.

    # ['merchant_category_code']:
    # This part of the code selects the 'merchant_category_code' column from the filtered DataFrame.
    # The result is a Series containing the merchant category codes for the relevant transactions.

    # .eq(merchant_category_code):
    # This part of the code compares the 'merchant_category_code' Series with the merchant_category_code variable.
    # The result is a new Series of boolean values, where True indicates that the merchant category code matches the merchant_category_code variable, and False indicates that it doesn't.

    # .all():
    # This part of the code checks if all the boolean values in the Series are True.
    # If all the values are True, it means that the 'merchant_category_code' for all the relevant transactions matches the merchant_category_code variable.

    # if not ... return False:
    # The entire expression is wrapped in an if not statement.
    # If the .all() check returns False, it means that not all the 'merchant_category_code' values match the merchant_category_code variable.
    # In this case, the code returns False, indicating that the condition is not met.



    if not transaction_window[transaction_window['card_id'] == card_id]['merchant_category_code'].eq(merchant_category_code).all():
        return False

    return True

def process_transaction(transaction):
    # Extract relevant fields from the transaction
    transaction_amount = float(transaction['transactionAmount'])
    card_balance = float(transaction['cardBalance'])
    transaction_time = datetime.datetime.strptime(transaction['dateTimeTransaction'], '%y%m%d%H%M%S')
    merchant_category_code = int(transaction['merchantCategoryCode'])
    latitude = float(transaction['latitude'])
    longitude = float(transaction['longitude'])
    location = (latitude, longitude)

    # Calculate transactional attributes
    transaction_age = datetime.datetime.now() - transaction_time
    transaction_age_hours = transaction_age.total_seconds() / 3600

    # Update transaction data
    transaction_data.append([transaction_amount, card_balance, transaction_age_hours, merchant_category_code, location])
    transaction_amounts.append(transaction_amount)
    transaction_locations.append(location)

    # Train the Isolation Forest model when enough data is available
    if len(transaction_data) >= model_params['i_forest']['n_samples']:
        X = np.array(transaction_data)
        i_forest.fit(X)

        # Predict outliers

        #  After training the Isolation Forest model, the code uses the i_forest.decision_function(X) method to calculate the outlier scores for each transaction in the X data.
        #  The np.where() function is used to identify the indices of the transactions that are considered outliers, based on the 'contamination' value specified in the model_params['i_forest'] dictionary.

        outlier_scores = i_forest.decision_function(X)
        outliers = np.where(outlier_scores > model_params['i_forest']['contamination'])[0]

        # Flag transactions based on the rules provided
        rule_violations = []

        for i in outliers:
            transaction_amount = X[i][0]
            card_balance = X[i][1]
            transaction_age_hours = X[i][2]
            merchant_category_code = X[i][3]
            location = X[i][4]

            if transaction_amount >= 0.7 * card_balance and card_balance >= 300000 and transaction_age_hours <= 12:
                rule_violations.append('RULE-001')

            if len(set(transaction_locations)) > 5 and np.min(np.diff([location[0] for location in transaction_locations])) >= 200000 and sum(transaction_amounts) >= 100000:
                rule_violations.append('RULE-002')

            if not is_coherent_pattern(transaction_amount, location, transaction_age_hours, card_balance, merchant_category_code):
                rule_violations.append('RULE-003')

            if not is_coherent_pattern_with_merchant_category_code(transaction_amount, location, transaction_age_hours, card_balance, merchant_category_code):
                rule_violations.append('RULE-004')

        return {
            'status': 'ALERT' if rule_violations else 'OK',
            'ruleViolated': rule_violations,
            'timestamp': str(int(time.time()))
        }

@app.route('/api/transaction', methods=['POST'])
def handle_transaction():
    transaction = request.get_json()
    response = process_transaction(transaction)
    return json.dumps(response)

if __name__ == '__main__':
    app.run(debug=True)