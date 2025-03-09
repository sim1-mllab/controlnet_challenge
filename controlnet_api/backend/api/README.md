Usage for fastAPI:

```bash
uvicorn --app-dir="backend/api" main:app --reload --workers 1 --host 0.0.0.0 --port 8000
```

Open application in browser under `http://127.0.0.1:8000/docs` and try it out.