
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import math 
import tensorflow as tf
import tensorflow_datasets as tfds #libreria para obtener datos de prueba
print("TensorFlow Version: ",tf.__version__)

#FUNCIONES
#NORMALIZAR LOS DATOS (PASAR DE 0-255 A 0-1)
def normalizar(imagenes, etiquetas):
    print(f"imagen: {imagenes}, etiquetas: {etiquetas}")
    print("--------------------------------------------------")
    imagenes = tf.cast(imagenes,tf.float32)
    print(f"imagen: {imagenes}, etiquetas: {etiquetas}")
    print("--------------------------------------------------")
    imagenes /= 255 #AQUI PASA DE 0-255 A 0-1
    print(f"imagen: {imagenes}, etiquetas: {etiquetas}")
    print("--------------------------------------------------")
    return imagenes,etiquetas





#DESCARGAR METADATOS
datos, metadatos = tfds.load("fashion_mnist",as_supervised=True, with_info=True)

#CARACTERISTICA DE LOS DATOS
print(metadatos)

#SEPARAR EN DATOS DE PRUEBA Y DE ENTRENAMIENTO
datos_entrenamiento, datos_pruebas = datos["train"], datos["test"]

#VER LOS DATOS 
nombre_clases = metadatos.features["label"].names
print("\nClases de ropa: ",nombre_clases)

#NORMALIZAR LOS DATOS CON LA FUNCION
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_pruebas = datos_pruebas.map(normalizar)

#AGREGAR LOS DATOS A LA RAM/CACHE(MEMORIA EN LUGAR DE DISCO, SE HACE EL ENTRENAMIENTO MAS RAPIDO)
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()

#MOSTRAR UNA IMAGEN DE LOS DATOS DE PRUEBA, LA PRIMERA
for imagen,etiqueta in datos_entrenamiento.take(1):
    break

imagen = imagen.numpy().reshape((28,28)) #REDIMENSIONAR

#dibujar la imagen
plt.figure()
plt.imshow(imagen,cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show

#MOSTRAR MAS DE UNA IMAGEN CON A ETIQUETA DE LOS DATOS
plt.figure(figsize=(10,10))
for i, (imagen,etiqueta) in enumerate(datos_entrenamiento.take(25)):
    image = imagen.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagen, cmap=plt.cm.binary)
    plt.xlabel(nombre_clases[etiqueta])
plt.show()

#CREAR MODELO DE RED NEURONAL
modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)), #28x28 pixeles con un canal en blanco y negro: imagenes, flatten se encarga de convertir la matris y aplastarla a una dimension de 784(28x28) neuronas
    tf.keras.layers.Dense(50,activation=tf.nn.relu),#capas ocultas 1
    tf.keras.layers.Dense(50,activation=tf.nn.relu),#capa oculta 2
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)#capa de salida, la funcion de activacion softman se usa en la capa de salida en las redes de clasificacion para asegurar que la suma de las neuronas de salida siempre de 1
])

#COMPILAR MODELO
modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

#PARA QUE HACER QUE LA RED NEURONAL ENTRENE MAS RAPIDO PUEDE HACER EL APRENDIZAJE POR LOTES, OSEA POR PARTES
TAMAÑO_LOTE = 32
num_ej_entrenamiento = metadatos.splits["train"].num_examples
num_ej_pruebas = metadatos.splits["test"].num_examples
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMAÑO_LOTE)
datos_pruebas = datos_pruebas.batch(TAMAÑO_LOTE)

#ENTRENAR LA RED NEURONAL
historial = modelo.fit(datos_entrenamiento, epochs=5, steps_per_epoch=math.ceil(num_ej_entrenamiento/TAMAÑO_LOTE))

#FUNCION DE PERDIDA
plt.xlabel("#EPOCA")
plt.ylabel("#Magnitud de Perdida")
plt.plot(historial.history["loss"])

#PREDICCIONES DE IMAGENES
for imagenes_prueba,etiquetas_prueba in datos_pruebas.take(1):
    imagenes_prueba = imagenes_prueba.numpy()
    etiquetas_prueba = etiquetas_prueba.numpy()
    predicciones = modelo.predict(imagenes_prueba)
plt.show()

def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
    arr_predicciones, etiqueta_real, img = arr_predicciones[i], etiquetas_reales[i], imagenes[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img[...,0], cmap=plt.cm.binary)

    etiqueta_prediccion = np.argmax(arr_predicciones)
    if etiqueta_prediccion == etiqueta_real:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(nombre_clases[etiqueta_prediccion],
                                    100*np.max(arr_predicciones),
                                    nombre_clases[etiqueta_real]),
                                    color=color)

def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
    arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    grafica = plt.bar(range(10), arr_predicciones, color="#777777")
    plt.ylim([0, 1]) 
    etiqueta_prediccion = np.argmax(arr_predicciones)
    
    grafica[etiqueta_prediccion].set_color('red')
    grafica[etiqueta_real].set_color('blue')


filas = 5
columnas = 5
num_imagenes = filas*columnas
plt.figure(figsize=(2*2*columnas, 2*filas))
for i in range(num_imagenes):
  plt.subplot(filas, 2*columnas, 2*i+1)
  graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
  plt.subplot(filas, 2*columnas, 2*i+2)
  graficar_valor_arreglo(i, predicciones, etiquetas_prueba)
plt.show()