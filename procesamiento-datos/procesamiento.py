#iportar paqueterias
import numpy as np
import h5py
import random
import os
import shutil

nombre_carpeta = "temp"

if os.path.exists(nombre_carpeta) and os.path.isdir(nombre_carpeta):
    shutil.rmtree(nombre_carpeta)
    os.mkdir(nombre_carpeta)
else:
    os.mkdir(nombre_carpeta)

os.mkdir("datos-procesados")
os.mkdir("datos-procesados/train")
os.mkdir("datos-procesados/test")

#Intercamcio de elemento por su posición en la tabla periodica
def posicion_en_tabla_periodica(elemento_quimico):
    tabla_periodica = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
        "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Ni": 27, "Co": 28,
        "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
        "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46,
        "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
        "Cs": 55, "Ba": 56, "La": 57, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
        "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94,
        "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100
    }

    elemento = elemento_quimico.capitalize()
    if elemento in tabla_periodica:
        return tabla_periodica[elemento]
    else:
        return None
    
# Nombre del archivo de entrada y salida

###################################################
archivo_entrada = "10pt-300-800K-tri-input.data" 
###################################################
    
#Encontrar número de nuestras
num_muestras=0
with open(archivo_entrada, "r") as f_in:
    for linea in f_in:
        linea=linea.split()
        if linea[0]=='begin':
            num_muestras=num_muestras+1

print(f"\nEl archivo tiene {num_muestras} muestras")

i=0
num_atomos=0
with open(archivo_entrada, "r") as f_in:
    for linea in f_in:
        linea=linea.split()
        if linea[0]=='begin':
            i=i+1
        if i==2:
            break
        if linea[0]=='atom':
            num_atomos=num_atomos+1

print(f"\nCada arreglo tiene {num_atomos} atomos")

print(f"\nDando un total de {num_muestras*num_atomos} atomos")

with open(archivo_entrada, "r") as f_in:
    for linea in f_in:
        linea=linea.split()
        if linea[0]=='begin':
            continue
        if linea[0]=='atom':
            n_pos=[linea[1],linea[2],linea[3]]
            z=posicion_en_tabla_periodica(linea[4])
            n_type=[z]
            n_force=[linea[7],linea[8],linea[9]]
            with open("temp/pos.data", "a") as file:
                line = " ".join(str(elemento) for elemento in n_pos)
                file.write(line + "\n")
            with open("temp/type.data", "a") as file:
                line = " ".join(str(elemento) for elemento in n_type)
                file.write(line + "\n")
            with open("temp/force.data", "a") as file:
                line = " ".join(str(elemento) for elemento in n_force)
                file.write(line + "\n")
        if linea[0]=='energy':
            n_energy=[linea[1]]
            with open("temp/energy.data", "a") as file:
                line = " ".join(str(elemento) for elemento in n_energy)
                file.write(line + "\n")
        if linea[0]=='charge':
            continue
        if linea[0]=='end':
            continue

print("\nDatos procesados")

# Leer los datos del archivo "energy.data"
with open("temp/energy.data", "r") as f:
    lines = f.readlines()
# Procesar los datos y crear un arreglo NumPy
data = []
for line in lines:
    values = line.strip().split()
    data.append([float(val) for val in values])
# Convertir la lista en un arreglo NumPy
energy = np.array(data)

# Leer los datos del archivo "type.data"
with open("temp/type.data", "r") as f:
    lines = f.readlines()
# Procesar los datos y crear un arreglo NumPy
data = []
for line in lines:
    values = line.strip().split()
    data.append([float(val) for val in values])
# Convertir la lista en un arreglo NumPy
types = np.array(data)

# Leer los datos del archivo "pos.data"
with open("temp/pos.data", "r") as f:
    lines = f.readlines()
# Procesar los datos y crear un arreglo NumPy
data = []
for line in lines:
    values = line.strip().split()
    data.append([float(val) for val in values])
# Convertir la lista en un arreglo NumPy
pos = np.array(data)

# Leer los datos del archivo "force.data"
with open("temp/force.data", "r") as f:
    lines = f.readlines()
# Procesar los datos y crear un arreglo NumPy
data = []
for line in lines:
    values = line.strip().split()
    data.append([float(val) for val in values])
# Convertir la lista en un arreglo NumPy
forces = np.array(data)

print("\narray's creados")

energy=np.reshape(energy,(num_muestras))
pos=np.reshape(pos,(num_muestras,num_atomos,3))
types=np.reshape(types,(num_muestras,num_atomos))
forces=np.reshape(forces,(num_muestras,num_atomos,3))

print("\nDimensiones acomodadas")

#Separación train, test

#Generar una selección de indices aleatoria

###################################
seed_value = 42####################
random.seed(seed_value)#############
####################################
num_test=int(0.05*num_muestras)#####
####################################
numeros = list(range(num_muestras))
muestra_aleatoria = random.sample(numeros, num_test)  
muestra_aleatoria.sort()
muestra_aleatoria=np.array(muestra_aleatoria)

energy_test=energy[muestra_aleatoria]
pos_test=pos[muestra_aleatoria]
types_test=types[muestra_aleatoria]
forces_test=forces[muestra_aleatoria]

energy=np.delete(energy,muestra_aleatoria, axis=0)
pos=np.delete(pos,muestra_aleatoria, axis=0)
types=np.delete(types,muestra_aleatoria, axis=0)
forces=np.delete(forces,muestra_aleatoria, axis=0)


#Crear archivos .npy

#train
np.save('datos-procesados/train/pos.npy',pos)
np.save('datos-procesados/train/z.npy',types)
np.save('datos-procesados/train/y.npy',energy)
np.save('datos-procesados/train/neg_dy.npy', forces)

#test
np.save('datos-procesados/test/pos_test.npy',pos_test)
np.save('datos-procesados/test/z_test.npy',types_test)
np.save('datos-procesados/test/y_test.npy',energy_test)
np.save('datos-procesados/test/neg_dy_test.npy', forces_test)

print("\nArchivos .npy creados")

#Crear archivos .h5

#train
########################################################
filename = "datos-procesados/train/10pt-300-800K-tri.h5"##
########################################################

f=h5py.File(filename, "w")

##############################
group_name = "10pt-300-800K-tri"##
##############################

group = f.create_group(group_name)

# Guardar datos en el grupo
group.create_dataset("types", data=types)
group.create_dataset("pos", data=pos)
group.create_dataset("energy", data=energy)
group.create_dataset("forces", data=forces)

f.close()

#test
############################################################
filename = "datos-procesados/test/10pt-300-800K-tri-test.h5"##
############################################################

f=h5py.File(filename, "w")

###################################
group_name = "10pt-300-800K-tri-test"##
###################################

group = f.create_group(group_name)

# Guardar datos en el grupo
group.create_dataset("types", data=types_test)
group.create_dataset("pos", data=pos_test)
group.create_dataset("energy", data=energy_test)
group.create_dataset("forces", data=forces_test)

f.close()

print("\nArchivos .h5 creados")

if os.path.exists(nombre_carpeta) and os.path.isdir(nombre_carpeta):
    shutil.rmtree(nombre_carpeta)