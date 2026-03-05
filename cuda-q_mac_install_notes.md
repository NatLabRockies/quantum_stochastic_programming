# Installing and Running CUDA-Q on Mac

Detailed install instructions [here](https://nvidia.github.io/cuda-quantum/latest/using/install/install.html).
Container information [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum).

## Installing Pre-Requisites

Agian from the documentation [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum), need one of the following to run the container:
* Singularity
* Docker Engine

<!-- Going to install singularity and try that as docker requires administrator privileges and Docker Desktop seems to require a paid subscription under certain circumstances (I think we could use it since it would be non-commercial but it is still a bit of a gray area). -->

<!-- It seems that installing/running singularity requires a virtual machine. Installing this software requires admin access. -->

### Install Docker

Available for mac os with Docker Desktop--install instructions [here](https://docs.docker.com/desktop/setup/install/mac-install/). However:

From Matt S: "You usually have to ask ITS to install it for you, giving a justification in the request, something to the effect of how "I'm a scientific software developer who needs Docker to build containers on my laptop, which is becoming an increasingly popular method of distributing software in my field"

Business case I used (tried being honest--see if it works): "Installation and use of CUDA-Q on a mac desktop requires Docker Desktop to run.  Without Docker Desktop, development of CUDA-Q based algorithms will be significantly slower."

### Install Singularity

Mac install instructions [here](https://docs.sylabs.io/guides/3.0/user-guide/installation.html#install-on-windows-or-mac).

Above instructions uses homebrew.

This also seems to require admin privileges.

## Installing CUDA-Q

Following instructions given [here](https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#install-docker-image).

Using Docker CLI, download docker image
```bash
docker pull nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest
```
and run from the command line with
```bash
docker run -it --name cuda-quantum nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest
```

Directions for using VSCode with the container are [here](https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#docker-in-vscode).

Run and be happy...
