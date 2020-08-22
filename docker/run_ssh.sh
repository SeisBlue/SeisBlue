docker container rm -f seisnn_ssh
docker run -d \
        --gpus all \
        -p 49154:22 \
        -p 8888:8888 \
        -p 6006:6006 \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/shadow:/etc/shadow:ro \
        -v /etc/group:/etc/group:ro \
        -v </path/to/seisnn>:/SeisNN \
        -v </path/to/workspace>:/home/${USER} \
        -v </path/to/sfile>:/home/${USER}/SFILE_ROOT \
        -v </path/to/database>:/home/${USER}/SDS_ROOT:ro \
        --name seisnn_ssh seisnn/dev