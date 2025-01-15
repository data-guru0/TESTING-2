from config.paths_config import *
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.model_training import ModelTraining
from utils.common_functions import read_yaml

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    data_processor = DataProcessing(input_file=ANIMELIST_CSV, output_dir=PROCESSED_DIR)
    data_processor.process_data()

    model_trainer = ModelTraining(config_path=CONFIG_PATH, data_path=PROCESSED_DIR)
    model_trainer.train_model()