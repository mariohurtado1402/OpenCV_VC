# 1) Elegimos la imagen base (Ubuntu 22.04)
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} AS base

# 2) Variables de entorno
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 3) Paquetes mínimos + habilitar 'universe'
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    gnupg2 \
    lsb-release \
    curl \
    ca-certificates \
    build-essential \
    git \
    wget \
    locales \
    python3-pip \
    x11-apps \
    software-properties-common \
  && rm -rf /var/lib/apt/lists/* \
  && add-apt-repository universe

# 4) Clave y repositorio de ROS 2 Humble
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
  && echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
     > /etc/apt/sources.list.d/ros2.list

# 5) Instalamos ROS 2 Humble + colcon/rosdep/vcstool + dependencias adicionales
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-jinja2 \
    python3-numpy \
    python3-yaml \
    python3-matplotlib \
    libtool \
    libxml2-dev \
    libxslt1-dev \
    libgeographic-dev \
    libopencv-dev \
    clang-12 \
    python-is-python3 \
    python3-wxgtk4.0 \
  && rm -rf /var/lib/apt/lists/*

# 6) Generar locale en_US.UTF-8
RUN locale-gen en_US.UTF-8

# 7) Creamos el usuario "user" con UID 1000 y sudo sin contraseña
ARG USER=user
ARG UID=1000
RUN useradd -m -u ${UID} -s /bin/bash ${USER} \
  && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USER}
WORKDIR /home/${USER}

# 8) Copiamos entrypoint.sh e imponemos permisos (si tienes un entrypoint.sh, copia este archivo al contenedor)
COPY --chown=${USER}:${USER} entrypoint.sh /home/${USER}/entrypoint.sh
RUN sudo chmod +x /home/${USER}/entrypoint.sh

# 9) Definimos variables de entorno para ROS2
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc

# 10) Directorio de trabajo donde montaremos nuestro código local
WORKDIR /home/${USER}/vc_ws

# 11) Punto de entrada y comando por defecto
ENTRYPOINT ["/home/${USER}/entrypoint.sh"]
CMD ["bash"]
