import json
import time
import datetime
import numpy as np
from flask import Flask, request
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

# Load model parameters from a JSON file
with open('model_params.json') as f:
    model_params = json.load(f)

# Initialize the Isolation Forest model
i_forest = IsolationForest(n_estimators=model_params['i_forest']['n_estimators'],
                           contamination=model_params['i_forest']['contamination'],
                           random_state=model_params['i_forest']['random_state'])

# Initialize variables to store transaction data
transaction_data = []
transaction_amounts = []
transaction_locations = []

def is_coherent_pattern():
    return 0

def is_coherent_pattern_with_merchant_category_code():
    return 0

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