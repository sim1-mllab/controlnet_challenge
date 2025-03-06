NOTE: docker image is not using any CUDA base image due to image size and simplicity of the use case.
STRONG assumption: CUDA is available on the device.

# Build Docker
```bash
docker build -t awesomedemo .
```

Run image:
```bash
docker run awesomedemo
```

# CONTRIBUTION
Install pre-commit with `pip add pre-commit` and initialize for this project with `pre-commit install`.
