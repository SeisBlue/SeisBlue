# Warning!!!! Please change the port "49154" below for security
docker container rm -f tf_ssh
nvidia-docker run -d -p 49154:22 --name tf_ssh tensorflow_ssh
