import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Carpeta de resultados ----------------------------------------

CARPETA_RESULTADOS = "Imagenes-Resultados"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)


def guardar_figura(nombre):
    plt.savefig(
        os.path.join(CARPETA_RESULTADOS, nombre + ".png"),
        dpi=150,
        bbox_inches="tight"
    )


# --- Cargo imagen --------------------------------------------------
img = cv2.imread(
    "Imagenes/Imagen_con_detalles_escondidos.tif",
    cv2.IMREAD_GRAYSCALE
)


# --- Exploración inicial -------------------------------------------

# Tipo de objeto
print("Tipo:", type(img))

# Dimensiones
print("Shape:", img.shape)

h, w = img.shape
print("Alto:", h, "pixeles")
print("Ancho:", w, "pixeles")

# Tipo de dato
print("Tipo de dato:", img.dtype)

# Cantidad total de píxeles
print("Cantidad de pixeles:", img.size)

# Valores mínimo y máximo
print("Intensidad minima:", img.min())
print("Intensidad maxima:", img.max())

# Valores de gris presentes
pix_vals = np.unique(img)

print("Cantidad de niveles de gris presentes:", len(pix_vals))
print("Valores de gris presentes:")
print(pix_vals)

# Estadísticas básicas
print("Intensidad media:", np.mean(img))
print("Desvio estandar:", np.std(img))


# --- Visualización -------------------------------------------------

plt.figure()
h = plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.colorbar(h)
plt.title("Imagen original")
guardar_figura("p1_imagen_original")
plt.show(block=True)

# --- Histograma de la imagen original -----------------------------

hist, bins = np.histogram(img.flatten(), 256, [0, 256])

plt.figure()
plt.plot(hist)
plt.title("Histograma de la imagen original")
plt.xlabel("Nivel de intensidad")
plt.ylabel("Cantidad de pixeles")
guardar_figura("p1_histograma_original")
plt.show(block=True)

# --- Prueba de ecualización en una ventana local ------------------

M = 31
N = 31

fila = 128
columna = 128

mitad_M = M // 2
mitad_N = N // 2

ventana = img[
    fila-mitad_M:fila+mitad_M+1,
    columna-mitad_N:columna+mitad_N+1
]

print("Tamaño de la ventana:", ventana.shape)
print("Valor del pixel central:", img[fila, columna])
print("Minimo local:", ventana.min())
print("Maximo local:", ventana.max())

plt.figure()
plt.imshow(ventana, cmap="gray", vmin=0, vmax=255)
plt.title("Ventana local 31 x 31")
plt.colorbar()
guardar_figura("p1_ventana_local_31x31")
plt.show(block=True)

# --- Histograma local ---------------------------------------------

hist_local, bins = np.histogram(ventana.flatten(), 256, [0, 256])

# Histograma normalizado
histn_local = hist_local.astype(np.double) / ventana.size

# CDF local
cdf_local = histn_local.cumsum()

# Valor original del píxel central
valor_central = img[fila, columna]

# Nuevo valor según la ecualización local
nuevo_valor = round(cdf_local[valor_central] * 255)

print("Valor original del pixel central:", valor_central)
print("CDF para ese valor:", cdf_local[valor_central])
print("Nuevo valor ecualizado:", nuevo_valor)

# --- Visualización del histograma local y CDF ---------------------

plt.figure()

plt.subplot(121)
plt.plot(hist_local)
plt.title("Histograma local")
plt.xlabel("Nivel de intensidad")
plt.ylabel("Cantidad de pixeles")

plt.subplot(122)
plt.plot(cdf_local)
plt.title("CDF local")
plt.xlabel("Nivel de intensidad")
plt.ylabel("Probabilidad acumulada")

guardar_figura("p1_histograma_cdf_local")
plt.show(block=True)

# --- Ecualización local de histograma -----------------------------

def ecualizacion_local(img, M, N):

    mitad_M = M // 2
    mitad_N = N // 2

    # Agrego bordes para poder procesar también los píxeles extremos
    img_pad = cv2.copyMakeBorder(
        img,
        mitad_M,
        mitad_M,
        mitad_N,
        mitad_N,
        cv2.BORDER_REPLICATE
    )

    # Imagen de salida
    img_out = np.zeros(img.shape, dtype=np.uint8)

    # Recorro todos los píxeles de la imagen original
    for fila in range(img.shape[0]):
        for columna in range(img.shape[1]):

            # Extraigo la ventana M x N
            ventana = img_pad[
                fila:fila + M,
                columna:columna + N
            ]

            # Histograma local
            hist, _ = np.histogram(
                ventana.flatten(),
                256,
                [0, 256]
            )

            # Histograma normalizado
            histn = hist.astype(np.double) / ventana.size

            # CDF local
            cdf = histn.cumsum()

            # Valor del píxel central
            valor = img[fila, columna]

            # Ecualización del píxel central
            img_out[fila, columna] = round(cdf[valor] * 255)

    return img_out

# --- Aplico ecualización local ------------------------------------

img_local = ecualizacion_local(img, 31, 31)

# --- Comparación ---------------------------------------------------

plt.figure()

ax1 = plt.subplot(121)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Imagen original")
plt.xticks([])
plt.yticks([])

plt.subplot(122, sharex=ax1, sharey=ax1)
plt.imshow(img_local, cmap="gray", vmin=0, vmax=255)
plt.title("Ecualización local 31 x 31")
plt.xticks([])
plt.yticks([])

guardar_figura("p1_original_vs_ecualizada_31x31")
plt.show(block=True)

# --- Comparación 2---------------------------------------------------

img_local_7 = ecualizacion_local(img, 7, 7)
img_local_15 = ecualizacion_local(img, 15, 15)
img_local_31 = ecualizacion_local(img, 31, 31)
img_local_63 = ecualizacion_local(img, 63, 63)

plt.figure()

plt.subplot(221)
plt.imshow(img_local_7, cmap="gray", vmin=0, vmax=255)
plt.title("Ventana 7 x 7")
plt.xticks([]), plt.yticks([])

plt.subplot(222)
plt.imshow(img_local_15, cmap="gray", vmin=0, vmax=255)
plt.title("Ventana 15 x 15")
plt.xticks([]), plt.yticks([])

plt.subplot(223)
plt.imshow(img_local_31, cmap="gray", vmin=0, vmax=255)
plt.title("Ventana 31 x 31")
plt.xticks([]), plt.yticks([])

plt.subplot(224)
plt.imshow(img_local_63, cmap="gray", vmin=0, vmax=255)
plt.title("Ventana 63 x 63")
plt.xticks([]), plt.yticks([])

guardar_figura("p1_comparacion_ventanas")
plt.show(block=True)

'''
Globalmente, la imagen posee un rango de intensidades amplio debido a la
diferencia entre el fondo claro y las regiones oscuras. Sin embargo, dentro
de una ventana local, el rango de intensidades puede ser mucho menor.

Por ejemplo, si en una ventana los valores van de 0 a 10 y el píxel central
tiene intensidad 8, globalmente ese valor sigue siendo muy bajo frente al
rango total de la imagen. Pero dentro de ese vecindario, 8 representa un
valor relativamente alto.

La ecualización local aprovecha esta información del entorno para aumentar
el contraste de cada píxel según los valores presentes en su propia ventana.
'''
