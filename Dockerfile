# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy all files from the current directory into the container
COPY . /app

# Install system dependencies from packages.txt
RUN apt-get update && xargs -a packages.txt apt-get install -y

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the dynamic port
EXPOSE $PORT

# Run the app with Streamlit
CMD ["streamlit", "run", "cml.py", "--server.port=$PORT", "--server.enableCORS=false", "--server.enableWebsocketCompression=false"]
