# Use Google Cloud SDK Alpine-based image
FROM google/cloud-sdk:alpine

# Set environment variables to prevent Python from writing .pyc files & Ensure Python output is not buffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python and necessary dependencies (pip, gfortran, git, etc.)
RUN apk add --no-cache python3 py3-pip libhdf5-dev libblas-dev liblapack-dev gfortran git

# Copy the application code into the container
COPY . /app

# Set working directory to /app
WORKDIR /app

# Install the required Python dependencies, including DVC
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install dvc

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable inside the Docker container
ARG GOOGLE_APPLICATION_CREDENTIALS
ENV GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}

# Authenticate with Google Cloud using the service account
RUN gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
RUN gcloud config set project ${GCP_PROJECT}

# Pull the data from DVC using gcloud authentication
RUN dvc pull

# Ensure the 'artifacts' directory and required subfolders exist
RUN mkdir -p artifacts/raw artifacts/processed artifacts/model artifacts/model_checkpoint artifacts/weights && chmod -R 777 artifacts/

# Optionally: Check if the 'animelist.csv' file exists in the 'artifacts/raw' folder
RUN test -f artifacts/raw/animelist.csv || (echo "File animelist.csv not found!" && exit 1)

# Train the model using the training pipeline script
RUN python pipeline/training_pipeline.py

# Expose the port that Flask will run on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "application.py"]
