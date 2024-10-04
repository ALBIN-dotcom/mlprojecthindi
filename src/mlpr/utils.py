import os
import sys
from src.mlpr.exception import CustomException
from src.mlpr.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

load_dotenv()

# Load environment variables
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv('db')

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        # Ensure environment variables are loaded
        if not all([host, user, password, db]):
            raise CustomException("Database connection parameters are not set in the environment variables.")

        with pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        ) as mydb:
            logging.info("Connection established: %s", mydb)
            df = pd.read_sql_query('SELECT * FROM student', mydb)
            print(df.head())
            return df
    except Exception as ex:
        raise CustomException(f"An error occurred: {ex}")
