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

RUN conda install -n music_analysis_env_final -c conda-forge -c pytorch -c nvidia \
    numpy=1.23.5 \
    pandas=1.5.3 \
    protobuf=3.20.3 \
    sdl \
    httpx \
    httpcore \
    tensorflow \
    pytorch=1.12.1 \
    cudatoolkit=11.6 \
    -y && \
    conda clean -afy

SHELL ["conda", "run", "-n", "music_analysis_env_final", "/bin/bash", "-c"]

# Copy requirements file for pip packages
COPY requirements.txt /app/requirements.txt

# Install pip packages, then remove pip's numpy and reinstall conda's version
RUN python -u -m pip install -vvv --no-cache-dir --progress-bar on -r /app/requirements.txt && \
    pip uninstall -y numpy && \
    conda install -n music_analysis_env_final numpy=1.23.5 -y

WORKDIR /app

ENTRYPOINT ["conda", "run", "-n", "music_analysis_env_final"]
CMD ["python", "driver.py"]