docker container rm -f seisblue_data
docker run -d \
        -p 49154:22 \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/shadow:/etc/shadow:ro \
        -v /etc/group:/etc/group:ro \
        -v </path/to/workspace>:/home/${USER} \
        -v </path/to/sfile>:/home/${USER}/SFILE_ROOT:ro \
        -v </path/to/database>:/home/${USER}/SDS_ROOT:ro \
        --name seisblue_ssh seisblue/data