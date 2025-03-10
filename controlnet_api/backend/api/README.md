# Usage for fastAPI:

## Pre-requisites:
For this app to run, you need to have ControlNet installed in the root of this repository, i.e. controlnet_api/ControlNet.
Either clone the repository or download the zip file and extract it in the root of this repository.

## Installation:
```bash
uvicorn --app-dir="backend/api" main:app --reload --workers 1 --host 0.0.0.0 --port 8000
```

Open application in browser under `http://127.0.0.1:8000/docs` and try it out.
