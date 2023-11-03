#!/bin/bash
echo "Iniciando el contenedor"
docker start torchmd-net-tesis
echo "Estas dentro del contenedor"
echo "Recuerda ejecutar exit para salir del contenedor"
echo "Recuerda ejecutar el script stop-docker para detener el contenedor"
docker exec -it torchmd-net-tesis /bin/bash

