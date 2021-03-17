# Nancy
The MLOps tools in docker

# Installation
First of all clone this repo

```bash
git clone https://github.com/PiePline/nancy.git
```

## Prepare for install
0. cd `nancy`
1. `mkdir env && cd env`
2. Create file s3.env that contains:
    2.1 MINIO_ACCESS_KEY=<your access key>
    2.2 MINIO_SECRET_KEY=<your secret key>
    2.3 S3_BUCKET=<your default bucket name>

## Setup Jenkins
Follow this [guide](https://www.jenkins.io/doc/tutorials/build-a-multibranch-pipeline-project/#setup-wizard)