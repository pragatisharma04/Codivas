import random
from datetime import datetime, timedelta

def generate_random_datetime():
    """
    Generates a random datetime in the "YYYYMMDDHMS" format.
    """
    # Generate random year, month, day, hour, minute, and second
    random_year = random.randint(1900, 2100)
    random_month = random.randint(1, 12)
    random_day = random.randint(1, 28)  # Assuming the month has at least 28 days
    random_hour = random.randint(0, 23)
    random_minute = random.randint(0, 59)
    random_second = random.randint(0, 59)
    random_millisecond = random.randint(0,9999)

    # Create a datetime object with the random values
    random_datetime = datetime(random_year, random_month, random_day, random_hour, random_minute, random_second, random_millisecond)

    # Format the datetime as a string in "YYYYMMDDHMS" format
    # datetime_str = random_datetime.strftime('%Y%m%d%H%M%S')

    return random_datetime