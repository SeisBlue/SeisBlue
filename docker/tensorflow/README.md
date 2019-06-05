# Tensorflow customized image

List of pre-installed python packages:
- tensorflow
- tfx 
- apache-airflow 
- docker
- obspy 
- scikit-learn 
- tqdm 

If you want to modify the [Dockerfile](Dockerfile), look up [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) for instructions.

Build the Docker image:

`./build_docker_image.sh`

Create workspace for home, will mount later into the container:

`mkdir workspace`

Run docker container, you can download the script [run.sh](run.sh).

change the followings: 
- 49154 for ssh port
- </path/to/database> for database or trace data folder
- </path/to/workspace> for home in the container

```
docker run -d \
    --runtime=nvidia \
    -p 49154:22 \
    -p 0.0.0.0:6006:6006 \
    -p 8080:8080 \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/shadow:/etc/shadow:ro \
    -v </path/to/database>:/mnt/DATA \
    -v </path/to/workspace>:/home/${USER} \
    --name tfx \
    tfx_ssh
```

Now you can SSH into the container with your username and password.

`ssh username@localhost -p49154`  

---

Remove container:

`docker container rm tfx`

Remove image:

`docker image rm tfx_ssh`

