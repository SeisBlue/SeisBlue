# Docker 
Nvidia-docker currently only works on Linux.

Install [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

You can pull the pre-built image from the dockerhub:

`docker pull jimmy60504/tfx_ssh`

Create workspace for home, will mount later into the container:

`mkdir workspace`

Run docker container, change the followings in [run.sh](docker/tensorflow/run.sh): 
- 49154 for ssh port
- </path/to/database> for database or trace data folder
- </path/to/workspace> for home in the container

Now you can SSH into the container with your username and password.

`ssh username@localhost -p49154`

---

Mounted volumes are data bridges between the host and containers.

Your data will stayed in the mounted volumes, any modifications will be destroyed when starting a new container.

Put your script in the workspace folder and you will find them in the container's home folder. 

---

Customized your Dockerfile in [docker/tensorflow](docker/tensorflow)

Remove all Docker container and images with [docker_clean_all.sh](docker/docker_clean_all)