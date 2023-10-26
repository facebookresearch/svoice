docker run --rm \
    --gpus all \
    -v `pwd`/:/workspace \
    --workdir=/workspace \
    -it doma945/temp