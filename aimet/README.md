
```
git clone https://github.com/quic/aimet.git
# tested with f931a169101e31e849a441422b9a8f7ff548bb92

# https://github.com/quic/aimet/blob/develop/packaging/docker_install.md#docker-information
# build docker image
cd aimet
docker build -t aimet_torch_gpu -f $PWD/Jenkins/Dockerfile.torch-gpu .
```