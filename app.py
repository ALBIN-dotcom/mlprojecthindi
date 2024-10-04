from src.mlpr.logger import logging
from src.mlpr.exception import CustomException
from src.mlpr.components.data_ingestion import DataIngestion
from src.mlpr.components.data_ingestion import DataIngestionConfig
import sys

if __name__=="__main__":
    logging.info("The execution has started")
    
    try:
       # data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info("custom Exception")
        raise CustomException(e,sys)
    