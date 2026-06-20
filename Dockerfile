# BioAnalysis platform image.
# NOTE: this base image is CPU-only. For GPU inference on EC2, either use an
# NVIDIA CUDA base image (e.g. nvidia/cuda:12.1.1-runtime-ubuntu22.04) and
# install Python on top, or run the systemd service from scripts/setup_ec2.sh
# directly on a GPU host. SAM3 inference on CPU is functional but very slow.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps: libgl1 + libglib2.0 for opencv/skimage, git for editable installs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install API/pipeline deps first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the whole project (expects submodules/MedSAM3 + weights to be present).
COPY . .

# Install the MedSAM3 (SAM3 + LoRA) package if it was vendored into the build.
RUN if [ -f submodules/MedSAM3/setup.py ]; then \
        pip install -e submodules/MedSAM3 ; \
    fi

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
