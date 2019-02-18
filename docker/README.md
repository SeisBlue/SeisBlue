# Docker 
Nvidia-docker currently only works on Linux.

Install [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

You can pull the pre-built image from the dockerhub:

`docker pull jimmy60504/pytorch_ssh`

`docker pull jimmy60504/tensorflow_ssh`

Once the image is pulled, run it with the `run.sh`, and you can ssh to your docker container.