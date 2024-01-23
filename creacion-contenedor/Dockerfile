# Usa una imagen base de Ubuntu
FROM emmanuelzula/tesis:0.1

# Documenta la intención de exponer el puerto 8888, pero no lo publica por sí mismo
# Recuerda usar -p al ejecutar el contenedor para mapear el puerto de host al contenedor
EXPOSE 8888/tcp

# Establece el directorio de trabajo en /workspace
WORKDIR /workspace

# Ejecuta un comando durante la construcción de la imagen
# Agrega una línea al final de ~/.bashrc para activar el ambiente virtual 'torchmd-net'
RUN /bin/sh -c echo "mamba activate torchmd-net" >> ~/.bashrc

# Comando por defecto al iniciar el contenedor
# Activa automáticamente el ambiente virtual al iniciar el contenedor
CMD ["bash"]
