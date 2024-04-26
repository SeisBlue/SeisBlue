# Dockerized Python Interpreter 

## Run in terminal:

```
docker run -it --rm \
        -v "$PWD":/usr/src/myapp \
        -w /usr/src/myapp \
        seisblue/data_python \
        python <test.py>
```
or

```
sh docker/run_python.sh <test.py>
```

## Run in PyCharm Professional Edition:

1. Add Python Interpreter with Docker form SSH

2. PyCharm | Preferences | Build, Execution & Deployment | Deployment | Mappings
  ```
  Local Path: <local project root>
  Deployment Path: <remote ssh project root>
  ```

3. Run | Edit Configurations | Docker container settings | Run options

  ```
  --rm -it -v <remote ssh project root>:/usr/src/app
  ```