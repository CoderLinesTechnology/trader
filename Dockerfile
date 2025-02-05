# Use a minimal Python image
FROM python:3.9-slim

# Set a working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install plotly matplotlib tensorflow


# Copy the rest of the application code
COPY . .

# Expose the port (Render will override this)
EXPOSE 8000

# Set environment variable for Render
ENV PORT=8000

# Run the bot (Modify this if you're using Flask, FastAPI, etc.)
CMD ["python", "main.py"]
