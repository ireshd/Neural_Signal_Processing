# Neural Signal DSP Pipeline - Docker Image
# Python environment for neural signal processing and visualization

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for matplotlib and scientific computing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create data directory for outputs
RUN mkdir -p data/spike_templates data/outputs

# Set matplotlib to use non-interactive backend by default
ENV MPLBACKEND=Agg

# Set Python to run in unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Default command - run the signal generator demo
CMD ["python", "src/signal_gen.py"]

