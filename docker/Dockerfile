# ====================
# Stage 1: Base
# ====================
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS base

ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    sudo \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1-mesa \
    libglib2.0-0 \
    x11-apps \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME} && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

COPY --chown=${USER_NAME}:${USER_NAME} docker/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

CMD ["/bin/bash"]

# ====================
# Stage 2: PerAct2
# ====================
FROM base AS peract2

ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-sync1 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

RUN pip3 install --no-cache-dir open3d

RUN wget -q https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz \
    && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz \
    && rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

ENV COPPELIASIM_ROOT=/home/${USER_NAME}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
ENV LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
ENV QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"

RUN git clone https://github.com/markusgrotz/PyRep.git \
    && cd PyRep \
    && pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir -e .

RUN git clone https://github.com/markusgrotz/RLBench.git \
    && cd RLBench \
    && pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir -e .

WORKDIR /home/${USER_NAME}

CMD ["/bin/bash"]