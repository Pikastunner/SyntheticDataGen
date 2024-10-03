FROM python:3.10-slim
WORKDIR /usr/src/app

# Copy requirements first to leverage Docker's caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code after the dependencies are installed
COPY . .

# Run the application
CMD ["python", "src/app.py"]
