# Start with Miniconda as the base image
FROM continuumio/miniconda3

ENV PATH_TMP=/app/maincode
ENV PYTHONPATH=$PATH_TMP/controlnet_api
ENV CONDA_DEFAULT_ENV=control

WORKDIR $PATH_TMP

# install system dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y unzip wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# clone repository and download the model
RUN wget https://github.com/lllyasviel/ControlNet/archive/ed85cd1e25a5ed592f7d8178495b4483de0331bf.zip -O ControlNet.zip &&  \
    unzip ControlNet.zip && \
    mv ControlNet-ed85cd1e25a5ed592f7d8178495b4483de0331bf ControlNet && \
    rm ControlNet.zip && \
    wget https://huggingface.co/lllyasviel/ControlNet/resolve/38a62cbf79862c1bac73405ec8dc46133aee3e36/models/control_sd15_canny.pth -P ControlNet/models && \
    rm -rf /root/.git && \
    rm -rf /var/lib/apt/lists/*

# create conda environment using the original environment.yaml file
RUN conda env create -f ControlNet/environment.yaml && \
    conda clean --all -f -y

# set the conda environment activation and path variables directly to avoid reliance on .bashrc
ENV PATH="/opt/conda/envs/control/bin:$PATH"

# copy API folder with backend and src/ code
COPY . $PATH_TMP/controlnet_api
EXPOSE 8080
CMD ["uvicorn", "backend.api.app:app", "--reload",  "--port",  "8080", "--host",  "0.0.0.0"]
