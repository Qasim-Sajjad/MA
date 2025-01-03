# Dockerfile
FROM ubuntu:22.04

ENV CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget --no-verbose https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init bash && \
    /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:$PATH
RUN conda --version

RUN conda create -n music_analysis_env_final python=3.10 pip -y && \
    conda clean -afy

RUN conda install -n music_analysis_env_final -c conda-forge \
    numpy=1.23.5 \
    pandas \
    protobuf=3.20.3 \
    sdl \
    httpx \
    httpcore \
    tensorflow \
    -y && \
    conda clean -afy

SHELL ["conda", "run", "-n", "music_analysis_env_final", "/bin/bash", "-c"]

# Copy requirements file for pip packages
COPY requirements.txt /app/requirements.txt

# Install pip packages using the requirements file
RUN python -u -m pip install --verbose --progress-bar on -r /app/requirements.txt

WORKDIR /app

ENTRYPOINT ["conda", "run", "-n", "music_analysis_env_final"]
CMD ["python", "driver.py"]