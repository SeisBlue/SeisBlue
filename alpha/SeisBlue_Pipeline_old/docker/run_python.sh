docker run -it --rm \
        -v "$PWD":/usr/src/myapp \
        -w /usr/src/myapp \
        seisblue/data_python \
        python "$@"