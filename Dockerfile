# Start with Miniconda as the base image
FROM continuumio/miniconda3

ENV PATH_TMP=/app/maincode
ENV PYTHONPATH=$PATH_TMP/controlnet_api
ENV CONDA_DEFAULT_ENV=control

WORKDIR $PATH_TMP

# install system dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# clone repository and download the model
RUN git clone https://github.com/lllyasviel/ControlNet.git && \
    mkdir -p ControlNet/models && \
    wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth -P ControlNet/models && \
    rm -rf /root/.git && \
    rm -rf /var/lib/apt/lists/*

# create conda environment using the original environment.yaml file
RUN conda env create -f ControlNet/environment.yaml && \
    conda clean --all -f -y

# set the conda environment activation and path variables directly to avoid reliance on .bashrc
ENV PATH="/opt/conda/envs/control/bin:$PATH"

# copy given Python script and JPG into git repo
COPY engineering/code/awesomedemo.py ControlNet
COPY engineering/img/mri_brain.jpg ControlNet/test_imgs/
# activate environment and run script
#WORKDIR $PYTHONPATH
#CMD ["bash", "-c", "source activate control && python awesomedemo.py"]

# copy API folder with backend and src/ code
COPY controlnet_api $PATH_TMP/controlnet_api
WORKDIR $PATH_TMP/controlnet_api
EXPOSE 8080
CMD ["uvicorn", "backend.api.app:app", "--reload",  "--port",  "8080", "--host",  "0.0.0.0"]
