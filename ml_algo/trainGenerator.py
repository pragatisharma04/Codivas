import csv
import random

import random
from datetime import datetime, timedelta
import dateTimeGenerator

# Function to generate random values for a row
def generate_random_row(sample_row):
    random_row = []
    for value in sample_row:
        if(value=="dateTimeTransaction"):
            random_row.append(dateTimeGenerator.generate_random_datetime())
        elif(value=="latitude"):
             random_row.append(random.randint(-90, 90))
        elif (value=="longitude"):
            random_row.append(random.randint(-180, 180))
        elif isinstance(value, int):
            random_row.append(random.randint(0, 1000000))
        elif isinstance(value, float):
            random_row.append(random.randint(0, 1000000))
        else:
            random_row.append(str(random.randint(0, 1000000)))
    return random_row

# Read the input CSV file
input_file = 'model_params.csv'
with open(input_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    # Get the sample row
    sample_row = next(reader)

# Write the output CSV file
output_file = 'train.csv'
num_rows = 10000  # Number of random rows to generate
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header row
    writer.writerow(sample_row)
    # Generate and write the random rows
    for _ in range(num_rows):
        random_row = generate_random_row(sample_row)
        writer.writerow(random_row)