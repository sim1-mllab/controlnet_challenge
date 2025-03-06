# Start with Miniconda as the base image
FROM continuumio/miniconda3

# Install system dependencies in a single RUN command to minimize layers
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Define the environment variable for the working directory
ENV PATH_TMP=/app/maincode
WORKDIR $PATH_TMP

# Clone the repository and download the model in a single RUN command to reduce layers
RUN git clone https://github.com/lllyasviel/ControlNet.git && \
    mkdir -p ControlNet/models && \
    wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth -P ControlNet/models && \
    rm -rf /root/.git && \
    rm -rf /var/lib/apt/lists/*

# Set PYTHONPATH environment variable
ENV PYTHONPATH=$PATH_TMP/ControlNet
WORKDIR $PYTHONPATH

# Create the conda environment using the provided environment.yaml file
RUN conda env create -f environment.yaml && \
    conda clean --all -f -y

# Set the conda environment activation and path variables directly to avoid reliance on .bashrc
ENV PATH="/opt/conda/envs/control/bin:$PATH"
ENV CONDA_DEFAULT_ENV=control

# Copy the Python script and image to the appropriate directories
COPY engineering/code/awesomedemo.py $PYTHONPATH/
COPY engineering/img/mri_brain.jpg $PYTHONPATH/test_imgs/

# Activate conda environment and run the Python script with the CMD command
CMD ["bash", "-c", "source activate control && python $PYTHONPATH/awesomedemo.py"]
