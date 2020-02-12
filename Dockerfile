FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

ARG project_dir=/app/
WORKDIR $project_dir
# ADD setup* $project_dir 
# ADD slowfast ${project_dir}/slowfast

ENV PYTHONPATH ${project_dir}/lib:$PYTHONPATH

RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y \
    git \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    gcc \
    vim \
 && pip install \
    numpy \
    torchvision==0.4.1 \
    sklearn \
    jupyter \
    pillow==6.2.2 \
    matplotlib \
    japanize_matplotlib \
    cloudpickle \
    pandas \
 && conda update -n base conda \
 && conda install av -c conda-forge \
 && conda install -c conda-forge opencv \
 && conda install -c conda-forge python-levenshtein \ 
 
