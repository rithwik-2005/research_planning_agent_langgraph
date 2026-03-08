# Base image: slim Python 3.11
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install dependencies FIRST (Docker caches this layer)
# So if only your .py files change, pip install is skipped on rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the backend source files
COPY blog_writer.py .
COPY main.py .

# Tell Docker this container listens on port 8000
EXPOSE 8000

# Command to start FastAPI when the container runs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]