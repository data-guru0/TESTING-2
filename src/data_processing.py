import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from src.custom_exception import CustomException
from src.logger import get_logger
from config.paths_config import *
import sys

# Initialize logger
logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir
        self.rating_df = None
        self.user2user_encoded = {}
        self.user2user_decoded = {}
        self.anime2anime_encoded = {}
        self.anime2anime_decoded = {}
        self.X_train_array = None
        self.X_test_array = None
        self.y_train = None
        self.y_test = None
        self.anime_df = None
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("DataProcessing instance initialized.")

    def load_data(self, usecols):
        """Loads the data from the CSV file."""
        try:
            self.rating_df = pd.read_csv(self.input_file, low_memory=True, usecols=usecols)
            logger.info(f"Data loaded successfully from {self.input_file}. Shape: {self.rating_df.shape}")
        except Exception as e:
            raise CustomException(f"Failed to load data from {self.input_file}", sys)

    def filter_users(self, min_ratings=400):
        """Filters out users with fewer than a specified number of ratings."""
        try:
            n_ratings = self.rating_df['user_id'].value_counts()
            self.rating_df = self.rating_df[self.rating_df['user_id'].isin(n_ratings[n_ratings >= min_ratings].index)].copy()
            logger.info(f"Filtered users with at least {min_ratings} ratings. Remaining data shape: {self.rating_df.shape}")
        except Exception as e:
            raise CustomException("Error during user filtering.", sys)

    def scale_ratings(self):
        """Scales the ratings to be between 0 and 1."""
        try:
            min_rating = self.rating_df['rating'].min()
            max_rating = self.rating_df['rating'].max()
            self.rating_df['rating'] = self.rating_df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values.astype(np.float64)
            logger.info(f"Ratings scaled between 0 and 1.")
        except Exception as e:
            raise CustomException("Error during rating scaling.", sys)

    def encode_data(self):
        """Encodes user IDs and anime IDs into numerical values."""
        try:
            user_ids = self.rating_df["user_id"].unique().tolist()
            self.user2user_encoded = {x: i for i, x in enumerate(user_ids)}
            self.user2user_decoded = {i: x for i, x in enumerate(user_ids)}
            self.rating_df["user"] = self.rating_df["user_id"].map(self.user2user_encoded)
            
            anime_ids = self.rating_df["anime_id"].unique().tolist()
            self.anime2anime_encoded = {x: i for i, x in enumerate(anime_ids)}
            self.anime2anime_decoded = {i: x for i, x in enumerate(anime_ids)}
            self.rating_df["anime"] = self.rating_df["anime_id"].map(self.anime2anime_encoded)
            logger.info("User and anime encoding completed.")
        except Exception as e:
            raise CustomException("Error during user and anime encoding.", sys)

    def split_data(self, test_set_size=1000, random_state=73):
        """Splits the data into training and testing sets."""
        try:
            self.rating_df = self.rating_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            X = self.rating_df[['user', 'anime']].values
            y = self.rating_df["rating"]
            train_indices = self.rating_df.shape[0] - test_set_size

            X_train, X_test, y_train, y_test = (
                X[:train_indices],
                X[train_indices:],
                y[:train_indices],
                y[train_indices:],
            )

            self.X_train_array = [X_train[:, 0], X_train[:, 1]]
            self.X_test_array = [X_test[:, 0], X_test[:, 1]]
            self.y_train = y_train
            self.y_test = y_test
            logger.info("Data split into training and testing sets.")
        except Exception as e:
            raise CustomException("Error during data splitting.", sys)

    def save_artifacts(self):
        """Saves encoded mappings and processed data for later use."""
        try:
            artifacts = {
                "user2user_encoded": self.user2user_encoded,
                "user2user_decoded": self.user2user_decoded,
                "anime2anime_encoded": self.anime2anime_encoded,
                "anime2anime_decoded": self.anime2anime_decoded,
            }
            for name, data in artifacts.items():
                joblib.dump(data, os.path.join(self.output_dir, f"{name}.pkl"))
                logger.info(f"{name} saved successfully.")

            joblib.dump(self.X_train_array, X_TRAIN_ARRAY)
            joblib.dump(self.X_test_array, X_TEST_ARRAY)
            joblib.dump(self.y_train, Y_TRAIN)
            joblib.dump(self.y_test, Y_TEST)
            
            # Save the rating DataFrame
            self.rating_df.to_csv(os.path.join(self.output_dir, "rating_df.csv"), index=False)
            logger.info("Processed data and rating DataFrame saved successfully.")
        except Exception as e:
            raise CustomException("Error during artifact saving.", sys)

    def process_anime_data(self):
        """Processes the anime data as per the given steps."""
        try:
            # Load anime data
            df = pd.read_csv(os.path.join('artifacts', 'raw', 'anime.csv'))
            cols = ["MAL_ID", "Name", "Genres", "sypnopsis"]
            sypnopsis_df = pd.read_csv(os.path.join('artifacts', 'raw', 'anime_with_synopsis.csv'),usecols=cols)
            df = df.replace("Unknown", np.nan)

            def getAnimeName(anime_id):
                try:
                    name = df[df.anime_id == anime_id].eng_version.values[0]
                    if pd.isna(name):
                        name = df[df.anime_id == anime_id].Name.values[0]
                except:
                    name = None
                return name

            df['anime_id'] = df['MAL_ID']
            df["eng_version"] = df['English name']
            df['eng_version'] = df.anime_id.apply(lambda x: getAnimeName(x))

            df.sort_values(by=['Score'], inplace=True, ascending=False, kind='quicksort', na_position='last')
            df = df[["anime_id", "eng_version", "Score", "Genres", "Episodes", "Type", "Premiered", "Members"]]

            # Save processed anime data
            df.to_csv(os.path.join(self.output_dir, 'anime_df.csv'), index=False)
            sypnopsis_df.to_csv(os.path.join(self.output_dir, 'synopsis_df.csv'), index=False)
            logger.info("Anime data processed and saved successfully.")
        except Exception as e:
            raise CustomException("Error during anime data processing.", sys)

    def process_data(self):
        """Executes the complete data processing pipeline."""
        try:
            self.load_data(usecols=["user_id", "anime_id", "rating"])
            self.filter_users()
            self.scale_ratings()
            self.encode_data()
            self.split_data()
            self.save_artifacts()
            self.process_anime_data()  # Process the anime data
            logger.info("Data processing pipeline completed successfully.")
        except CustomException as e:
            logger.error(str(e))
            raise
        except Exception as e:
            raise CustomException("Error in data processing pipeline.", sys)


# Example usage:
if __name__ == "__main__":
    try:
        data_processor = DataProcessing(input_file=ANIMELIST_CSV, output_dir=PROCESSED_DIR)
        data_processor.process_data()
    except CustomException as e:
        logger.error(str(e))
    except Exception as e:
        logger.error("Unhandled exception occurred.", exc_info=True)
