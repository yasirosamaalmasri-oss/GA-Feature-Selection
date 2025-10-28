FROM python:3.13-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ ./src/

# Set the working directory to where main.py is located
WORKDIR /app/src

# Command to run the application
CMD ["python", "main.py"]
