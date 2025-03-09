This repository hosts a REST API that allows to generate images from text and images with ControlNet. 

# Requirements
- hardware: CUDA is required
- conda/miniconda installed for controlling the python environment
  - `ControlNet/` needs to be copied here
  
  ```bash
    wget https://github.com/lllyasviel/ControlNet/archive/ed85cd1e25a5ed592f7d8178495b4483de0331bf.zip -O ControlNet.zip 
    unzip ControlNet.zip
    mv ControlNet-ed85cd1e25a5ed592f7d8178495b4483de0331bf ControlNet 
    rm ControlNet.zip
    ``` 
- pre-trained model from hugging face needs to be downloaded and stored in `ControlNet/models/`:
    
    ```bash
    wget https://huggingface.co/lllyasviel/ControlNet/resolve/38a62cbf79862c1bac73405ec8dc46133aee3e36/models/control_sd15_canny.pth -P ControlNet/models
    ```

  - create environment:
  ```bash
    conda env create -f environment . yaml
    conda activate control  
    ```

To start the app, `cd controlnet_api` and follow instructions in the README.md there.