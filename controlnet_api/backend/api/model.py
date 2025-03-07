"""
FastAPI app code for endpoint queries/.
"""

from pathlib import Path

import imageio
from PIL import Image
from io import BytesIO
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from backend.schemas.base import GenerationParams
from backend.api.utils.model import model_train, store_image
from src.model import controlnet_orchestration as orcas
from src.utils.logging_utils import get_logger

logger = get_logger(__file__)

model_router = APIRouter(
    prefix="/model",
    tags=["model"],
    responses={404: {"description": "Not found"}},
)

UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PRE_TRAINED_MODEL = None

if PRE_TRAINED_MODEL is None:
    logger.info(
        "Loading pre-trained model - this is generally loaded as it will be re-used"
    )
    PRE_TRAINED_MODEL = orcas.load_model(
        model_file_path=orcas.MODEL_PTH_PATH, model_config_path=orcas.MODEL_CONFIG_PATH
    )


@model_router.get("/health")
async def health_check():
    if PRE_TRAINED_MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")
    return {"status": "Model is ready!"}


@model_router.post("/generate", response_class=JSONResponse)
async def upload_image(
    file: UploadFile = File(...), model_parameters: GenerationParams = Depends()
):
    if PRE_TRAINED_MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")
    try:
        logger.info(f"Reading uploaded file: {file.filename}")
        file_content = await file.read()
        logger.info(f"Read {len(file_content)} bytes from {file.filename}")

        file_location = UPLOAD_DIR / file.filename
        logger.info(f"Store uploaded data to {file_location}")
        with open(file_location, "wb") as f:
            f.write(file_content)

        # Open the image from the bytes data
        image = imageio.imread(BytesIO(file_content))  # Read image directly from bytes

        # load parameters
        parameters = model_parameters.dict()
        logger.info("apply the model")
        output_images = model_train(
            input_image=image, model=PRE_TRAINED_MODEL, params=parameters
        )
        logger.info("done!")

        logger.info("save the output image into a BytesIO object to return")
        counter = 0
        for res in output_images:
            counter += 1
            pil_image = Image.fromarray(res)
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)  # Go to the beginning of the BytesIO buffer
            output_filename = f"result_{counter}.png"
            store_image(image=pil_image, filepath=RESULT_DIR, filename=output_filename)

        # Return the output image as a StreamingResponse
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI()
    uvicorn.run(app, host="0.0.0.0", port=8000)
