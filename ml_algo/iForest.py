import json
import time
import datetime
import numpy as np
import pandas as pd
from flask import Flask, request
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import pickle 

# app = Flask(__name__)

# Load model parameters from a CSV file
model_params = pd.read_csv('model_params.csv')

# n_estimators=model_params['i_forest']['n_estimators']: This sets the number of trees in the Isolation Forest model. The n_estimators parameter determines the number of decision trees that will be used in the ensemble. The value for this parameter is taken from the 'n_estimators' key in the 'i_forest' dictionary of the model_params dictionary.
# contamination=model_params['i_forest']['contamination']: This sets the expected proportion of outliers in the dataset. The contamination parameter specifies the proportion of outliers in the dataset, which is used to determine the threshold for identifying anomalies. The value for this parameter is taken from the 'contamination' key in the 'i_forest' dictionary of the model_params dictionary.
# random_state=model_params['i_forest']['random_state']: This sets the random state or seed for the Isolation Forest model. The random_state parameter ensures that the model's results are reproducible by setting a specific seed for the random number generator. The value for this parameter is taken from the 'random_state' key in the 'i_forest' dictionary of the model_params dictionary.
# i_forest = IsolationForest(...): This line creates an instance of the Isolation Forest model with the specified parameters and assigns it to the variable i_forest.

n_estimators=100
max_samples='auto'
contamination=0.04
max_feature=1.0
n_jobs=-1
random_state=1

# Create the iforest object
iforest = IsolationForest(n_estimators=100, max_samples='auto',
contamination=0.04, max_features=1.0,
bootstrap=False, n_jobs=-1, random_state=1)


# Initialize variables to store transaction data
transaction_data = []
transaction_amounts = []
transaction_locations = []

def is_transaction_fraudulent(transaction_amount, transactions_df, time_window='all'):
    """
    Checks if a transaction is potentially fraudulent based on the coherence of the transaction pattern.
    
    Args:
        transaction_amount (float): The amount of the transaction to be checked.
        transactions_df (pandas.DataFrame): A DataFrame containing the transaction history.
        time_window (str, optional): The time window to consider. Can be 'all', '12-hour', '1-day', or '7-day'. Defaults to 'all'.
        
    Returns:
        bool: True if the transaction is potentially fraudulent, False otherwise.
    """
    # Convert the 'Date' column to datetime
    transactions_df['dateTimeTransaction'] = pd.to_datetime(transactions_df['dateTimeTransaction'])
    
    # Calculate the transaction windows
    now = datetime.now()
    twelve_hour_window = now - timedelta(hours=12)
    one_day_window = now - timedelta(days=1)
    seven_day_window = now - timedelta(days=7)
    
    # Check the coherence of the transaction pattern for the specified window
    if time_window == '12-hour':
        twelve_hour_transactions = transactions_df[(transactions_df['dateTimeTransaction'] >= twelve_hour_window) & (transactions_df['dateTimeTransaction'] <= now)]
        if len(twelve_hour_transactions) > 0 and transaction_amount > 3 * twelve_hour_transactions['transactionAmount'].mean():
            return True
    elif time_window == '1-day':
        one_day_transactions = transactions_df[(transactions_df['dateTimeTransaction'] >= one_day_window) & (transactions_df['dateTimeTransaction'] <= now)]
        if len(one_day_transactions) > 0 and transaction_amount > 2 * one_day_transactions['transactionAmount'].mean():
            return True
    elif time_window == '7-day':
        seven_day_transactions = transactions_df[(transactions_df['dateTimeTransaction'] >= seven_day_window) & (transactions_df['dateTimeTransaction'] <= now)]
        if len(seven_day_transactions) > 0 and transaction_amount > 1.5 * seven_day_transactions['transactionAmount'].mean():
            return True
    else:
        # Check all time windows
        if is_transaction_fraudulent(transaction_amount, transactions_df, '12-hour'):
            return True
        elif is_transaction_fraudulent(transaction_amount, transactions_df, '1-day'):
            return True
        elif is_transaction_fraudulent(transaction_amount, transactions_df, '7-day'):
            return True
    
    return False

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
    if transaction_amount >= 0.7 * card_balance and transaction_amount >= 300000:
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

def is_coherent_pattern_with_merchant_category_code(transaction_amount, transactions_df, time_window='all'):
        """
        Checks if a transaction is potentially fraudulent based on the coherence of the transaction pattern.
        
        Args:
            transaction_amount (float): The amount of the transaction to be checked.
            transactions_df (pandas.DataFrame): A DataFrame containing the transaction history.
            time_window (str, optional): The time window to consider. Can be 'all', '12-hour', '1-day', or '7-day'. Defaults to 'all'.
            
        Returns:
            bool: True if the transaction is potentially fraudulent, False otherwise.
        """
        # Convert the 'Date' column to datetime
        transactions_df['dateTimeTransaction'] = pd.to_datetime(transactions_df['dateTimeTransaction'])
        
        # Calculate the transaction windows
        now = datetime.now()
        thirty_day_window = now - timedelta(hours=30)
        three_day_window = now - timedelta(days=3)
        seven_day_window = now - timedelta(days=7)
        
        # Check the coherence of the transaction pattern for the specified window
        if time_window == '30-day':
            twelve_hour_transactions = transactions_df[(transactions_df['dateTimeTransaction'] >= thirty_day_window) & (transactions_df['dateTimeTransaction'] <= now)]
            if len(twelve_hour_transactions) > 0 and transaction_amount > 3 * twelve_hour_transactions['transactionAmount'].mean():
                return True
        elif time_window == '3-day':
            three_day_transactions = transactions_df[(transactions_df['dateTimeTransaction'] >= three_day_window) & (transactions_df['dateTimeTransaction'] <= now)]
            if len(three_day_transactions) > 0 and transaction_amount > 2 * three_day_transactions['transactionAmount'].mean():
                return True
        elif time_window == '7-day':
            seven_day_transactions = transactions_df[(transactions_df['dateTimeTransaction'] >= seven_day_window) & (transactions_df['dateTimeTransaction'] <= now)]
            if len(seven_day_transactions) > 0 and transaction_amount > 1.5 * seven_day_transactions['transactionAmount'].mean():
                return True
        else:
            # Check all time windows
            if is_transaction_fraudulent(transaction_amount, transactions_df, '30-day'):
                return True
            elif is_transaction_fraudulent(transaction_amount, transactions_df, '3-day'):
                return True
            elif is_transaction_fraudulent(transaction_amount, transactions_df, '7-day'):
                return True
        
        return False

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

def process_transaction():
        X = np.genfromtxt('train.csv',delimiter=',',skip_header=1,usecols=(2,3,18,10,45,46))
        # Apply the iforest object on the numpy ndarray X to create pred
        # pred is a numpy ndarray that returns 1 for inliers, -1 for outliers
        pred = iforest.fit_predict(X)

        # Extract outliers
        outlier_index = np.where(pred==-1)
        print("Outliers in training data")
        print(outlier_index)

        # Real time use
        Xrt = np.genfromtxt('test.csv',delimiter=',',skip_header=1,usecols=(2,3,18,10,45,46))
        yrt=iforest.fit_predict(Xrt)
        print(yrt)
        
        rule_violations = []

        for i in outlier_index:
            transaction_amount = X[i][2]
            card_balance = X[i][18]
            transaction_age_hours = X[i][2]
            card_id=X[i][16]

            current_time = datetime.now()

            

            merchant_category_code = X[i][10]
            location = (X[i][-2],X[i][-1])

            for i in range(len(transaction_amount)):
                # Convert the datetime to hours
                hours = current_time.hour + (current_time.minute / 60) + (current_time.second / 3600) + (current_time.microsecond / 3600000000)

                transaction_age_h = hours - transaction_age_hours[i]

                if (transaction_amount[i] >= 0.7 * card_balance[i] ) and card_balance[i] >= 300000 and transaction_age_h<=12 or card_balance[i]<=transaction_amount[i]:
                    rule_violations.append('RULE-001')
                    break

                if len(set(transaction_locations)) > 5 and np.min(np.diff([location[0] for location in transaction_locations])) >= 200000 and sum(transaction_amounts) >= 100000:
                    rule_violations.append('RULE-002')

                if not is_transaction_fraudulent(transaction_amount,model_params):
                    rule_violations.append('RULE-003')

                if not is_coherent_pattern_with_merchant_category_code(transaction_amount,model_params):
                    rule_violations.append('RULE-004')

        retval={
            'status': 'ALERT' if rule_violations else 'OK',
            'ruleViolated': rule_violations,
            'timestamp': str(int(time.time()))
        }

        print(retval)



process_transaction()
pickle.dump(iforest,open('model.pkl','wb'))


# @app.route('/api/transaction', methods=['POST'])
# def handle_transaction():
#     # This retrieves the JSON data from the incoming HTTP request and assigns it to the transaction variable.
#     transaction = request.get_json()
#     response = process_transaction(transaction)
#     return json.dumps(response)

# if __name__ == '__main__':
#     app.run(debug=True)