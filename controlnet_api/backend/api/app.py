"""
This is the main file for the FastAPI server. It contains the FastAPI app and the main function to run the server.
"""

import uvicorn
from fastapi import FastAPI

from backend.api import model
from src.utils.logging_utils import get_logger


logger = get_logger(__file__)

description = """
ToDo: FastAPI to allow for image generation with ControlNet via POST requests.
"""

app = FastAPI(
    title="MLEngineering Assignment",
    description=description,
    version="0.0.1",
)

app.include_router(model.model_router)

if __name__ == "__main__":
    logger.info("Starting the FastAPI server")
    uvicorn.run(
        "app:app",
        app_dir="backend/api/",
        reload=True,
        # workers=1,
        host="0.0.0.0",
        port=8000,
    )
