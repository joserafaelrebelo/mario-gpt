#!/bin/bash

OFF='\033[0m'
RED='\033[0;31m'
GRN='\033[0;32m'
BLU='\033[0;34m'

BOLD=$(tput bold)
NORM=$(tput sgr0)

ERR="${RED}${BOLD}"
RRE="${NORM}${OFF}"


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

IMAGE_NAME="rafaeljose/super_mario_levels"
IMAGE_TAG="v0"

# Check if NVIDIA runtime is available in Docker
CHECK_RUNTIME=$(docker info --format '{{.Runtimes.nvidia}}')

if [ "$CHECK_RUNTIME" = "<no value>" ]; then
  echo -e "Running with ${BLU}Integrated Graphics support runtime${OFF}"
  RUNTIME=' --devices /dev/dri '
else
  echo -e "Running with ${GRN}NVIDIA runtime${OFF}"
  RUNTIME=' --nvidia '
fi

if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "$IMAGE_NAME:$IMAGE_TAG"; then
    echo -e "${BOLD}Running simulation with image ${BLU}${BOLD}$IMAGE_NAME${OFF} ${BOLD} with tag ${BLU}${BOLD}$IMAGE_TAG${OFF}${BOLD}.${RRE}"
else
  echo -e "${ERR}ERROR: ${OFF}Docker image ${BLU}$IMAGE_NAME${OFF} with tag ${BLU}$IMAGE_TAG${OFF} not found!${RRE}"
  exit 
fi

sleep 2

rocker $RUNTIME --x11 \
  --name mario_gpt \
  --network host \
  --volume=../mario-gpt/:/mario-gpt:rw \
  $IMAGE_NAME:$IMAGE_TAG bash