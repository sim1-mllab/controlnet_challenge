NOTE: docker image is not using any CUDA base image due to image size and simplicity of the use case.
STRONG assumption: CUDA is available on the device.

# Build Docker
```bash
docker build -t controlnet_api .
```

Run image (replace <MY_IMAGE_NAME> with any name of your choice)
```bash
docker run --name <MY_IMAGE_NAME> --gpus all -it controlnet_api
```




# Clean Up
It makes sense to have an overview over existing images and containers and clean up on a regular basis to reduce space occupied by these object.

See all images that were built:
```bash
docker images
```

To get an overview over all containers (also stopped ones):
```bash
docker ps -a
```
Remove images with
```bash
docker rm <IMAGE_NAME>
```



# CONTRIBUTION
Install pre-commit with `pip add pre-commit` and initialize for this project with `pre-commit install`.
