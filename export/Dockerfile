FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    git curl build-essential libgl1-mesa-glx libglib2.0-0
WORKDIR /workspace
RUN python3.10 -m venv /opt/hailo_env
ENV PATH="/opt/hailo_env/bin:$PATH"
CMD ["/bin/bash"]