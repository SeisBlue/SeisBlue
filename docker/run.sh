docker container rm -f tf_ssh
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
        --name tf_ssh seisnn/tf_ssh