#!/bin/bash
set -e

# Esta variable la paso en 'docker run' con -e USE_GPU=true/false
if [ "${USE_GPU}" = "true" ]; then
  echo "🌟 Entrando en MODO GPU dentro del container"
  if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  No se detecta runtime NVIDIA. Asegúrate de ejecutar con '--gpus all' y de tener instalado NVIDIA Container Toolkit en el host."
  fi
else
  echo "🤖 Entrando en MODO CPU dentro del container"
fi

# Finalmente, ejecuta el comando que se pase (p. ej. bash)
exec "$@"
