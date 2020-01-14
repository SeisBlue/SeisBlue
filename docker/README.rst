Docker
======

Nvidia-docker currently only works on Linux.

Install `docker`_ and `nvidia-docker`_.

You can pull the pre-built image from the dockerhub:

``docker pull jimmy60504/tf_ssh``

Clone this repo:

``git clone https://github.com/jimmy60504/SeisNN.git``

Create workspace for home, will mount later into the container:

``mkdir workspace``

Run docker container, change the followings in `run.sh`_:

-  49154 for ssh port
-  </path/to/seisnn> for this project folder
-  </path/to/workspace> for home in the container
-  </path/to/sfile> for S-file folder
-  </path/to/database> for database or trace data folder

Now you can SSH into the container with your username and password.

``ssh username@localhost -p49154``

--------------

Mounted volumes are data bridges between the host and containers.

Your data will stayed in the mounted volumes, any modifications will be
destroyed when starting a new container.

Put your script in the workspace folder and you will find them in the
container's home folder.

--------------

Customized your Dockerfile in `docker/tensorflow`_

Remove all Docker container and images with `docker_clean_all.sh`_

Tensorflow customized image
===========================

List of pre-installed python packages:

-  tensorflow
-  docker
-  obspy
-  scikit-learn
-  tqdm

If you want to modify the `Dockerfile`_, look up `Best practices for
writing Dockerfiles`_ for instructions.

Build the Docker image:

``./build_docker_image.sh``

Clone this repo:

``git clone https://github.com/jimmy60504/SeisNN.git``

Create workspace for home, will mount later into the container:

``mkdir workspace``

Run docker container, you can download the script `run.sh <run.sh>`__.

change the followings:

-  49154 for ssh port
-  </path/to/seisnn> for this project folder
-  </path/to/workspace> for home in the container
-  </path/to/sfile> for S-file folder
-  </path/to/database> for database or trace data folder

::

   docker run -d \
           --gpus all \
           -p 49154:22 \
           -p 0.0.0.0:6006:6006 \
           -v /etc/passwd:/etc/passwd:ro \
           -v /etc/shadow:/etc/shadow:ro \
           -v /etc/group:/etc/group:ro \
           -v </path/to/seisnn>:/SeisNN \
           -v </path/to/workspace>:/home/${USER} \
           -v </path/to/sfile>:/mnt/sfile \
           -v </path/to/database>:/mnt/SDS_ROOT:ro \
           --name tf_ssh tf_ssh

Now you can SSH into the container with your username and password.

``ssh username@localhost -p49154``

--------------

Remove container:

``docker container rm tf``

Remove image:

``docker image rm tf_ssh``

.. _docker: https://docs.docker.com/install/linux/docker-ce/ubuntu/
.. _nvidia-docker: https://github.com/NVIDIA/nvidia-docker
.. _run.sh: tensorflow/run.sh
.. _docker/tensorflow: tensorflow
.. _docker_clean_all.sh: docker_clean_all
.. _Dockerfile: Dockerfile
.. _Best practices for writing Dockerfiles: https://docs.docker.com/develop/develop-images/dockerfile_best-practices/