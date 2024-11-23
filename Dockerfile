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

# Expose the port (Streamlit typically uses 8501, but Render will assign a dynamic port)
EXPOSE 8501

# Run the app with Streamlit, using Render's assigned $PORT
CMD streamlit run cml.py --server.port=8501 --server.address=0.0.0.0 --server.enableCORS=false --server.enableWebsocketCompression=false

