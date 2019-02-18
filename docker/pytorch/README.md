# Docker 
Nvidia-docker currently only works on Linux.

Install [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Download the Dockerfile to the server.

Change the root password in the Dockerfile

Build docker image:

`docker build -t pytorch_ssh .`

Run docker container, change the port "49154":

`nvidia-docker run -d -p 49154:22 --name pt_ssh pytorch_ssh`

Remove container:

`docker container rm -f pt_ssh`

Remove image:

`docker image rm pytorch_ssh`

---

You can pull the pre-built image from the dockerhub:

`docker pull jimmy60504/pytorch_ssh`

The root password is ssh.