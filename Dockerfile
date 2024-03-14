# Use the official lightweight Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir streamlit pandas joblib scikit-learn seaborn matplotlib xgboost

# Expose the port number that the app runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "--server.port", "8501", "app.py"]