import einops
import random
import torch
from pytorch_lightning import seed_everything
import numpy as np
import sys
from pathlib import Path
from src.utils.logging_utils import get_logger

# Get the parent directory of the current file (main.py)
parent_dir = Path(__file__).resolve().parents[3]
controlnet_dir = parent_dir / "ControlNet"

# Add the ControlNet folder to sys.path
sys.path.append(str(controlnet_dir))
# ControlNet imports
import config  # noqa: E402
from annotator.util import resize_image, HWC3  # noqa: E402
from annotator.canny import CannyDetector  # noqa: E402
from cldm.model import create_model, load_state_dict  # noqa: E402
from cldm.ddim_hacked import DDIMSampler  # noqa: E402


logger = get_logger(__file__)


MODEL_PTH_PATH = controlnet_dir / "models/control_sd15_canny.pth"
MODEL_CONFIG_PATH = controlnet_dir / "models/cldm_v15.yaml"


def load_model(model_file_path, model_config_path):
    model = create_model(config_path=model_config_path).cpu()
    model.load_state_dict(load_state_dict(model_file_path, location="cuda"))

    return model.cuda()


def process(
    model,
    input_image,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    image_resolution,
    ddim_steps,
    guess_mode,
    strength,
    scale,
    seed,
    eta,
    low_threshold,
    high_threshold,
):
    # original code from awesomedemo/process()
    # NEWLY ADDED - make dependencies explicit to function
    logger.info("Initialize models")
    ddim_sampler = DDIMSampler(model)
    apply_canny = CannyDetector()
    logger.info("Initialize models")
    ###

    with torch.no_grad():
        logger.info("resize image")
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        logger.info("set up controls")
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        logger.info("set seed")
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        logger.info("Apply control scales to model")
        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        logger.info("Get results")
        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results
