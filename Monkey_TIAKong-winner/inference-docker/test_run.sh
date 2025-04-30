#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_TAG="tiakong"
DOCKER_NOOP_VOLUME="${DOCKER_TAG}-volume"

INPUT_DIR="/media/u1910100/data/Monkey/test/input"
OUTPUT_DIR="/media/u1910100/data/Monkey/test/output"
MODEL_DIR="/home/u1910100/GitHub/Monkey_TIAKong/weights"


echo "=+= Cleaning up any earlier output"
if [ -d "$OUTPUT_DIR" ]; then
  # Ensure permissions are setup correctly
  # This allows for the Docker user to write to this location
  rm -rf "${OUTPUT_DIR}"/*
  chmod -f o+rwx "$OUTPUT_DIR"
else
  mkdir -m o+rwx "$OUTPUT_DIR"
fi


echo "=+= (Re)build the container"
docker build /home/u1910100/GitHub/Monkey_TIAKong \
  -f inference-docker/dockerfile \
  --platform=linux/amd64 \
  --tag $DOCKER_TAG 2>&1


echo "=+= Doing a forward pass"
## Note the extra arguments that are passed here:
# '--network none'
#    entails there is no internet connection
# 'gpus all'
#    enables access to any GPUs present
# '--volume <NAME>:/tmp'
#   is added because on Grand Challenge this directory cannot be used to store permanent files
docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null
docker run --rm \
    --platform=linux/amd64 \
    --network none \
    --gpus all \
    --volume "$INPUT_DIR":/input:ro \
    --volume "$OUTPUT_DIR":/output \
    --volume "$MODEL_DIR":/opt/ml/model:ro \
    --volume "$DOCKER_NOOP_VOLUME":/tmp \
    $DOCKER_TAG
docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null

# Ensure permissions are set correctly on the output
# This allows the host user (e.g. you) to access and handle these files
docker run --rm \
    --quiet \
    --env HOST_UID=`id --user` \
    --env HOST_GID=`id --group` \
    --volume "$OUTPUT_DIR":/output \
    alpine:latest \
    /bin/sh -c 'chown -R ${HOST_UID}:${HOST_GID} /output'

echo "=+= Wrote results to ${OUTPUT_DIR}"

echo "=+= Save this image for uploading via save.sh \"${DOCKER_TAG}\""