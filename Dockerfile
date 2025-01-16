# Use a lightweight Python image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files & Ensure Python output is not buffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies required for TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code
COPY . .

# Install the package in editable mode along with TensorFlow
RUN pip install --no-cache-dir -e .

# Create the artifacts directory before running the training pipeline
RUN mkdir -p artifacts

# Train the model before running the application
RUN python pipeline/training_pipeline.py

# Expose the port that Flask will run on
EXPOSE 5000

# Command to run the app
CMD ["python", "application.py"]
