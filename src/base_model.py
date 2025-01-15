from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense, BatchNormalization, Activation
from utils.common_functions import read_yaml  # Importing read_yaml from your utils directory
from src.custom_exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

class BaseModel:
    def __init__(self, config_path):
        try:
            self.config = read_yaml(config_path)  # Using your existing read_yaml method
            logger.info(f"Successfully loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            raise CustomException(f"Error loading configuration from {config_path}", e)
    
    def RecommenderNet(self, n_users, n_animes):
        """Defines the recommender model."""
        try:
            embedding_size = self.config['model']['embedding_size']
            
            # Input layers
            user = Input(name='user', shape=[1])
            anime = Input(name='anime', shape=[1])
            
            # Embedding layers
            user_embedding = Embedding(
                name='user_embedding',
                input_dim=n_users, 
                output_dim=embedding_size
            )(user)
            
            anime_embedding = Embedding(
                name='anime_embedding',
                input_dim=n_animes, 
                output_dim=embedding_size
            )(anime)
            
            # Dot product for similarity
            x = Dot(name='dot_product', normalize=True, axes=2)([user_embedding, anime_embedding])
            x = Flatten()(x)
            
            # Dense layer
            x = Dense(1, kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation("sigmoid")(x)
            
            # Compile model
            model = Model(inputs=[user, anime], outputs=x)
            model.compile(
                loss=self.config['model']['loss'], 
                optimizer=self.config['model']['optimizer'], 
                metrics=self.config['model']['metrics']
            )
            
            logger.info("Model created successfully")
            return model
        
        except Exception as e:
            logger.error(f"Error occurred while creating the model: {str(e)}")
            raise CustomException("Failed to create recommender model", e)
