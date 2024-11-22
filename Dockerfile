# Use an official Python runtime as a parent image
#FROM python:3.9-slim
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Streamlit configuration for larger file uploads

# Copy the rest of the application code into the container at /app
COPY . /app

# Define environment variable to indicate production environment
ENV FLASK_ENV=production

# Run the application
EXPOSE 8504

# Command to run the Streamlit app
#CMD ["streamlit", "run", "app.py", "--server.port=8503", "--server.enableCORS=false"]
CMD ["streamlit", "run", "ui.py", "--server.port=8504"]