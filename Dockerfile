# Start from a standard, slim Python 3.10 image
FROM python:3.10-slim

# [THE FIX] Install system dependencies for OpenCV
# Use 'libgl1' instead of 'libgl1-mesa-glx'
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's cache
COPY requirements.txt .

# Install all of your Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now, copy your ENTIRE project into the container
COPY . .

# Make our startup script executable
RUN chmod +x ./start.sh

# Expose the two ports our app uses:
# 8000 for the FastAPI backend
EXPOSE 8000
# 8501 for the Streamlit frontend
EXPOSE 8501

# This is the command that will run when the container starts
CMD ["./start.sh"]