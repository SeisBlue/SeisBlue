# Docker

# Docker 
Nvidia-docker currently only works on Linux.

Install [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Clone this repo:

`git clone https://github.com/SeisNN/SeisNN.git`

Create workspace for home, will mount later into the container:

`mkdir workspace`

Run docker container, change the followings in `run.sh`: 
- `49154` for ssh port
- `</path/to/seisnn>` for this project folder
- `</path/to/workspace>` for home in the container
- `</path/to/sfile>` for S-file folder
- `</path/to/database>` for database or trace data folder

Now you can SSH into the container with your username and password.  

    ssh username@localhost -p49154

## Data transfer between host and container

Mounted volumes are data bridges between the host and containers.

Your data will stayed in the mounted volumes, any modifications will be destroyed when starting a new container.

Put your scripts in the workspace folder and you will find them in the container's home folder. 

## Useful docker commands

Update image:

    docker pull seisnn/tf_ssh

Remove container:

    docker container rm tf_ssh

Remove image:

    docker image rm seisnn/tf_ssh

Remove all containers(!):

    docker rm -f $(docker ps -a -q)

Remove all images(!):

    docker rmi -f $(docker images -q)