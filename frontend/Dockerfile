# Same base but this image will be much smaller
# because requirements are minimal
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install ONLY frontend dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the frontend source file
COPY streamlit_app.py .

# Tell Docker this container listens on port 8501
EXPOSE 8501

# Command to start Streamlit when the container runs
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true"]