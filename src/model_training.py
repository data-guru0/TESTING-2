import joblib
import numpy as np
import os
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from utils.common_functions import read_yaml
from src.custom_exception import CustomException
from src.logger import get_logger
from src.base_model import BaseModel  # Importing BaseModel
from config.paths_config import *
import comet_ml  # Import comet_ml for experiment tracking


logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, config_path, data_path):
        """Initialize with configuration and data path."""
        self.config = read_yaml(config_path)
        self.data_path = data_path
        
        # Initialize Comet experiment for online mode
        self.experiment = comet_ml.Experiment(
            api_key="uqgrnGhGvBA0zC3HfdmGf2WN9",
            project_name="mlops-2",
            workspace="data-guru0" # Set your workspace name
        )
        logger.info("Comet experiment initialized in online mode.")
    
    def load_data(self):
        """Load training and testing data from .pkl files."""
        try:
            X_train_array = joblib.load(X_TRAIN_ARRAY)
            X_test_array = joblib.load(X_TEST_ARRAY)
            y_train = joblib.load(Y_TRAIN)
            y_test = joblib.load(Y_TEST)

            logger.info("Successfully loaded data")
            return X_train_array, X_test_array, y_train, y_test
        except Exception as e:
            logger.error("Error while loading data")
            raise CustomException("Failed to load data", e)

    def train_model(self):
        """Main method to train the model."""
        X_train_array, X_test_array, y_train, y_test = self.load_data()

        n_users = len(joblib.load(USER2USER_ENCODED))  # Loaded from processed data
        n_animes = len(joblib.load(ANIME2ANIME_ENCODED))  # Loaded from processed data

        # Initialize model using BaseModel
        base_model = BaseModel(config_path=CONFIG_PATH)
        model = base_model.RecommenderNet(n_users=n_users, n_animes=n_animes)  # Adjust n_users and n_animes as needed
        model.summary()

        # Learning rate schedule and callbacks
        start_lr = 0.00001
        min_lr = 0.00001
        max_lr = 0.00005
        batch_size = 10000
        rampup_epochs = 5
        sustain_epochs = 0
        exp_decay = .8

        def lrfn(epoch):
            if epoch < rampup_epochs:
                return (max_lr - start_lr) / rampup_epochs * epoch + start_lr
            elif epoch < rampup_epochs + sustain_epochs:
                return max_lr
            else:
                return (max_lr - min_lr) * exp_decay**(epoch - rampup_epochs - sustain_epochs) + min_lr

        lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=0)

        model_checkpoints = ModelCheckpoint(filepath=CHECKPOINT_FILE_PATH, 
                                            save_weights_only=True, 
                                            monitor='val_loss', 
                                            mode='min', 
                                            save_best_only=True)
        
        early_stopping = EarlyStopping(patience=3, 
                                       monitor='val_loss', 
                                       mode='min', 
                                       restore_best_weights=True)

        # Create necessary directories for saving models and weights
        os.makedirs(os.path.dirname(CHECKPOINT_FILE_PATH), exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(WEIGHTS_DIR, exist_ok=True)

        # Model training
        try:
            history = model.fit(
                x=X_train_array,
                y=y_train,
                batch_size=batch_size,
                epochs=20,
                verbose=1,
                validation_data=(X_test_array, y_test),
                callbacks=[model_checkpoints, lr_callback, early_stopping]
            )

            model.load_weights(CHECKPOINT_FILE_PATH)
            logger.info("Model training completed successfully")

            # Log the metrics for the training process in Comet
            for epoch in range(len(history.history['loss'])):
                train_loss = history.history['loss'][epoch]
                val_loss = history.history['val_loss'][epoch]
                logger.info(f"Epoch {epoch + 1}: Training Loss = {train_loss}, Validation Loss = {val_loss}")
                self.experiment.log_metric('train_loss', train_loss, step=epoch)
                self.experiment.log_metric('val_loss', val_loss, step=epoch)

                # If you have accuracy metrics, log them as well
                if 'accuracy' in history.history:
                    train_accuracy = history.history['accuracy'][epoch]
                    val_accuracy = history.history['val_accuracy'][epoch]
                    logger.info(f"Epoch {epoch + 1}: Training Accuracy = {train_accuracy}, Validation Accuracy = {val_accuracy}")
                    self.experiment.log_metric('train_accuracy', train_accuracy, step=epoch)
                    self.experiment.log_metric('val_accuracy', val_accuracy, step=epoch)

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise CustomException("Model training failed", e)

        self.save_model_and_weights(model)

    def save_model_and_weights(self, model):
        """Save model and extract weights."""
        try:
            # Save model
            model.save(MODEL_PATH)
            logger.info(f"Model saved successfully to {MODEL_PATH}")

            # Extract and normalize weights
            user_weights = self.extract_weights('user_embedding', model)
            anime_weights = self.extract_weights('anime_embedding', model)

            # Save weights using joblib
            joblib.dump(user_weights, WEIGHTS_PATH_USER)
            joblib.dump(anime_weights, WEIGHTS_PATH_ANIME)

            logger.info("User and anime weights saved successfully")

            # Log model saving event to Comet
            self.experiment.log_asset(MODEL_PATH)
            self.experiment.log_asset(WEIGHTS_PATH_USER)
            self.experiment.log_asset(WEIGHTS_PATH_ANIME)

        except Exception as e:
            logger.error(f"Error saving model or weights: {str(e)}")
            raise CustomException("Error saving model or weights", e)

    def extract_weights(self, layer_name, model):
        """Extract and normalize weights from a layer.""" 
        try:
            weight_layer = model.get_layer(layer_name)
            weights = weight_layer.get_weights()[0]
            weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))  # Normalize
            return weights
        except Exception as e:
            logger.error(f"Error extracting weights from {layer_name}: {str(e)}")
            raise CustomException(f"Error extracting weights from {layer_name}", e)


# Example usage
if __name__ == "__main__":
    model_trainer = ModelTraining(config_path=CONFIG_PATH, data_path=PROCESSED_DIR)
    model_trainer.train_model()
