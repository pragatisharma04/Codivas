import random
from datetime import datetime, timedelta

def generate_random_datetime():
    """
    Generates a random number of hours passed from a reference point.
    """
   
    random_hour = random.randint(0,100000000)
    return random_hour