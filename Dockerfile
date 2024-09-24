# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies
RUN apt update \
    && apt install -y libgl1-mesa-glx

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Write the environment variables to JSON files for Google Cloud Vision and Firebase
RUN echo $GOOGLE_APPLICATION_CREDENTIALS_JSON > /app/google-vision-config.json \
    && echo $FIREBASE_APPLICATION_CREDENTIALS_JSON > /app/firebase-config.json

# Copy the rest of the application files into the container
COPY . .

RUN pip uninstall -y opencv-python
# Expose the port that the app will run on
EXPOSE 8080

# Define environment variable for the port (optional)
ENV PORT=8080

# Run Uvicorn server, explicitly using the environment variable
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
