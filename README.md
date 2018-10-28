# TGS Salt Identification [Kaggle Competition]

## Installation
### Requirements

* Docker
* nvidia-docker to run on gpu

### Building docker image

```
git clone https://github.com/Laurans/TGS_Salt_Identification
cd TGS_Salt_Identification
docker build -t workspace_kaggle_bash .
```

This docker image will give you an bash access to the docker container to run `python3 <file>` command

## Usage