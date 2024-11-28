Running mario-gpt environment with Docker
===

This implementation uses a [super_mario_levels](https://hub.docker.com/r/rafaeljose/super_mario_levels) Docker image already available in Docker Hub, for speeding the process with no needs to build it from scratch.

## There are two possible ways of running it:

### - Docker Compose:
> This approach may not use 100% of your GPU to render things.

1. First you should allow xhost usage for root (as it is the user wich the container will run under). This step should be executed every time you restarted your computer
    ```sh
    xhost +local:root
    ```
2. Up the compose (`-d` is for detaching from the application):
    ```sh
    docker compose -f docker/docker-compose.yaml up -d
    ```
3. Now you must be able to access the *jupyter-lab* via:  
    http://localhost:8888/lab


### - Rocker script:
> This approach is more reliable in terms of efficiency of your computer resources

1. Make sure you have rocker installed correctly. Visit [rocker's official repo](https://github.com/osrf/rocker) for more info.

2. Simply run the bash script
    ```sh
    ./run_mario_env.sh
    ```