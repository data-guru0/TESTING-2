FROM google/cloud-sdk:alpine

# Set environment variables to prevent Python from writing .pyc files & Ensure Python output is not buffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python and necessary dependencies
RUN apk add --no-cache python3 py3-pip

# Install system dependencies required for TensorFlow and DVC
RUN apk add --no-cache libhdf5-dev libblas-dev liblapack-dev gfortran git

# Copy the application code
COPY . /app

# Install the required Python dependencies, including DVC
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install dvc

# Pull the data from DVC
RUN gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
RUN gcloud config set project ${GCP_PROJECT}
RUN dvc pull

# Ensure the 'artifacts' directory and required subfolders exist
RUN mkdir -p artifacts/raw artifacts/processed artifacts/model artifacts/model_checkpoint artifacts/weights && chmod -R 777 artifacts/

# Train the model before running the application
RUN python pipeline/training_pipeline.py

# Expose the port that Flask will run on
EXPOSE 5000

# Command to run the app
CMD ["python", "application.py"]
