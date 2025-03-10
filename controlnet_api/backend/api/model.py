"""
FastAPI app code for endpoint queries/.
"""

import zipfile
import imageio
from PIL import Image
from io import BytesIO
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from starlette.responses import StreamingResponse, JSONResponse, Response

from backend.api.utils import controlnet_orchestration as orcas
from backend.schemas.base import GenerationParams
from backend.api.utils.model import model_train, is_valid_image, ALLOWED_MIME_TYPES
from backend.api.utils.logging_utils import get_logger

logger = get_logger(__file__)

model_router = APIRouter(
    prefix="/model",
    tags=["model"],
    responses={404: {"description": "Not found"}},
)

PRE_TRAINED_MODEL = None
if PRE_TRAINED_MODEL is None:
    logger.info(
        "Loading pre-trained model - this is generally loaded as it will be re-used"
    )
    PRE_TRAINED_MODEL = orcas.load_model(
        model_file_path=orcas.MODEL_PTH_PATH, model_config_path=orcas.MODEL_CONFIG_PATH
    )
    logger.info("Model is loaded")


@model_router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint to verify if the model is ready.
    """
    if PRE_TRAINED_MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")
    return {"status": "Model is ready!"}


@model_router.post("/generate", response_class=StreamingResponse)
async def upload_image(
    file: UploadFile = File(...), model_parameters: GenerationParams = Depends()
) -> Response:
    """
    POST endpoint to generate images using the ControlNet model.
    file: The image file to be used as input for the model.
    model_parameters: The parameters to be used for the model.
    """
    if PRE_TRAINED_MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")

    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPG, JPEG, and PNG are allowed.",
        )

    try:
        logger.info(f"Reading uploaded file: {file.filename}")
        file_content = await file.read()
        logger.info("Check if the file is a valid image.")
        if not is_valid_image(file_content):
            raise HTTPException(
                status_code=400, detail="Uploaded file is not a valid image."
            )

        logger.info(f"Read {len(file_content)} bytes from {file.filename}")

        logger.info("Open the image from the bytes data")
        image = imageio.imread(BytesIO(file_content))

        parameters = model_parameters.dict()
        logger.info(f"load parameters {parameters}")
        logger.info("apply the model")
        output_images = model_train(
            input_image=image, model=PRE_TRAINED_MODEL, params=parameters
        )
        logger.info("done!")

        logger.info("save the output image into a BytesIO object to return")
        counter = 0
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            logger.info("Store original image")
            zip_file.writestr(file.filename, file_content)
            logger.info("Store generated images")
            for res in output_images:
                counter += 1
                pil_image = Image.fromarray(res)
                img_byte_arr = BytesIO()
                pil_image.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)  # Go to the beginning of the BytesIO buffer
                output_filename = f"result_{counter}.png"
                zip_file.writestr(output_filename, img_byte_arr.read())
        zip_buffer.seek(0)
        # Return the output image as a StreamingResponse
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=generated_images.zip"
            },
        )

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})
