# Docker 
Nvidia-docker currently only works on Linux.

Install [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Download the Dockerfile to the server.

Change the root password in the Dockerfile

Build docker image:

`docker build -t tensorflow_ssh .`

Run docker container, change the port "49154":

`nvidia-docker run -d -p 49154:22 --name tf_ssh tensorflow_ssh`

Remove container:

`docker container rm tf_ssh`

Remove image:

`docker image rm tensorflow_ssh`

---

You can pull the pre-built image from the dockerhub:

`docker pull jimmy60504/tensorflow_ssh`

The root password is ssh.