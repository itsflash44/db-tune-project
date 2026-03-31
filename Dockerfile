FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Copy the requirements and install them first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the environment files
COPY . .

# Expose the Hugging Face default port
EXPOSE 7860

# Command to run the OpenEnv FastAPI server
CMD ["python", "-m", "server.app"]