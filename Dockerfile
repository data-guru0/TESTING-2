# Use a lightweight Python image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files & Ensure Python output is not buffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies required for TensorFlow and DVC
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code
COPY . .

# Install the required Python dependencies, including DVC
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install dvc

# Add this line at the top to define the build argument
ARG GOOGLE_APPLICATION_CREDENTIALS

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable inside the Docker container
ENV GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}

# Pull the data from DVC
RUN dvc pull

# Ensure the 'artifacts' directory and required subfolders exist
RUN mkdir -p artifacts/raw artifacts/processed artifacts/model artifacts/model_checkpoint artifacts/weights && chmod -R 777 artifacts/

# Check if the 'animelist.csv' file is present in the 'artifacts/raw' folder before running the pipeline
RUN test -f artifacts/raw/animelist.csv || (echo "File animelist.csv not found!" && exit 1)

# Train the model before running the application
RUN python pipeline/training_pipeline.py

# Expose the port that Flask will run on
EXPOSE 5000

# Command to run the app
CMD ["python", "application.py"]
