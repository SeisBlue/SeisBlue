docker container rm -f tfx
docker run -d \
    --runtime=nvidia \
	-p 49154:22 \
	-p 0.0.0.0:6006:6006 \
	-p 8080:8080 \
	-v /etc/passwd:/etc/passwd:ro \
	-v /etc/shadow:/etc/shadow:ro \
	-v /etc/group:/etc/group:ro \
	-v </path/to/database>:/mnt/DATA \
	-v </path/to/workspace>:/home/${USER} \
	--name tfx tfx_ssh