"""
FastAPI app code for endpoint queries/.
"""

import matplotlib.pyplot as plt
# from datetime import datetime
# from typing import List, Optional

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path

from starlette.responses import StreamingResponse

from backend.utils.logging_utils import get_logger
from PIL import Image
from io import BytesIO

# from sqlalchemy.orm import Session
#
# from backend.api.utils.queries import get_queries, store_dicts_as_json
# from backend.db.db_setup import get_db
# from backend.schemas.tables import ArxivQuery

logger = get_logger(__file__)

model_router = APIRouter(
    prefix="/model",
    tags=["model"],
    responses={404: {"description": "Not found"}},
)


UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# This function will simulate model training using the uploaded image.
def model_train(input_image: Image):
    # Here you would implement your model training logic, such as loading the image
    # and training a model based on it.

    # shape of loaded the array
    logger.info(f"Image Size: {input_image.size}")  # (width, height)
    logger.info(f"Image Mode: {input_image.mode}")  # 'RGB', 'L', etc.
    logger.info(f"Image Format: {input_image.format}")
    plt.imshow(input_image)

    return input_image


def store_image(image: Image, filepath: Path, filename: str, format="PNG"):
    """
    Store image to folder, make sure folder exists
    """
    filepath.mkdir(parents=True, exist_ok=True)
    logger.info(f"save the image into {filepath}")
    image.save(filepath / filename, format=format)


@model_router.post("/", response_class=JSONResponse)
async def upload_image(file: UploadFile = File(...)):
    try:
        logger.info(f"Reading uploaded file: {file.filename}")
        file_content = await file.read()

        logger.info(f"Read {len(file_content)} bytes from {file.filename}")

        file_location = UPLOAD_DIR / file.filename
        logger.info(f"Store uploaded data to {file_location}")
        with open(file_location, "wb") as f:
            f.write(file_content)

        # Open the image from the bytes data
        image = Image.open(BytesIO(file_content))  # Read image directly from bytes
        image.show()

        logger.info("apply the model")
        output_image = model_train(image)
        logger.info("done!")

        logger.info("save the output image into a BytesIO object to return")
        img_byte_arr = BytesIO()
        output_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)  # Go to the beginning of the BytesIO buffer

        output_filename = "result.png"
        store_image(image=output_image, filepath=RESULT_DIR, filename=output_filename)

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
