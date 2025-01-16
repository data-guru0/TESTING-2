import os
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_names = self.config["bucket_file_names"]
        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"Data Ingestion started with {self.bucket_name} and files are {', '.join(self.file_names)}")

    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)

            # Loop through the files to download them
            for file_name in self.file_names:
                file_path = os.path.join(RAW_DIR, file_name)
                
                # If the file is 'animelist.csv', we download only the first 50 lakh rows
                if file_name == "animelist.csv":
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)
                    
                    # Read only the first 50 lakh rows from 'animelist.csv'
                    logger.info(f"Large file {file_name} downloaded, now reading the first 50 lakh rows.")
                    data = pd.read_csv(file_path, nrows=500)  # Read only 5 million rows
                    
                    # Save the first 50 lakh rows to the same file path
                    data.to_csv(file_path, index=False)
                    logger.info(f"50 Lakh rows of {file_name} saved to {file_path}")

                else:
                    # For other files, download them entirely
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)
                    logger.info(f"CSV file '{file_name}' successfully downloaded to {file_path}")

        except Exception as e:
            logger.error("Error while downloading the CSV files")
            raise CustomException("Failed to download CSV files", e)

    def run(self):
        try:
            logger.info("Starting data ingestion process")

            self.download_csv_from_gcp()

            logger.info("Data ingestion completed successfully")

        except CustomException as ce:
            logger.error(f"CustomException : {str(ce)}")

        finally:
            logger.info("Data ingestion completed")


if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
