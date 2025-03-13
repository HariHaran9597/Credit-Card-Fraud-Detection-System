# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Copy the project files
COPY . .

# Install dependencies and the project in development mode
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Create a script to run both services
RUN echo '#!/bin/bash\n\
streamlit run src/dashboard/app.py --server.port 8501 & \
uvicorn src.api.app:app --host 0.0.0.0 --port 8000' > start.sh

# Make the script executable
RUN chmod +x start.sh

# Run the services
CMD ["./start.sh"]