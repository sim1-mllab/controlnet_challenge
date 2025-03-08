from pathlib import Path

from PIL import Image
from src.model import controlnet_orchestration as orcas
from src.utils.logging_utils import get_logger

logger = get_logger(__file__)


def model_train(input_image, params: dict, model):
    # General information of image
    logger.info(f"Image size is {input_image.shape}")
    logger.info(f"Image dtype is {input_image.dtype}")

    # returns list of images, dependent on num_samples in args
    logger.info("Run image generation")
    result = orcas.process(model=model, input_image=input_image, **params)
    logger.info("Done with image generation")

    return result


def store_image(image: Image, filepath: Path, filename: str, format="PNG"):
    """
    Store image to folder, make sure folder exists
    """
    filepath.mkdir(parents=True, exist_ok=True)
    logger.info(f"save the image into {filepath}")
    image.save(filepath / filename, format=format)
