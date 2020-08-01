# Installation instructions

## Prerequisite:

- S-File catalog from [SEISAN](http://seisan.info/)
- SeisComP Data Structure (SDS) database. The directory and file layout of SDS is defined as:

      SDSROOT/YEAR/NET/STA/CHAN.TYPE/NET.STA.LOC.CHAN.TYPE.YEAR.DAY

## Installation:

- Follow the instructions in the [Docker](docker.md) folder to create a Docker container.
- SSH into the Docker container you create.

      ssh username@localhost -p49154

- Copy `/SeisNN/jupyter.sh` to your workspace and execute to start jupyter lab server

      cp /SeisNN/jupyter.sh ~/.
      chmod 777 jupyter.sh
      ./jupyter.sh

- Copy `/SeisNN/notebook` to your workspace

      cp -r /SeisNN/notebook ~/.

- Paste the URL with generate token into your local browser

      http://127.0.0.1:8888/?token=36b31a373a9d18cc9b30a50883ad5a3638b19bed47be8074
