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


# --- Cargo el examen ----------------------------------------------

img = cv2.imread(
    "Imagenes/examen_1.png",
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
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Examen 2")
plt.colorbar()
guardar_figura("p2_examen_original")
plt.show(block=True)

# --- Umbralado -----------------------------------------------------

T_otsu, img_th = cv2.threshold(
    img,
    thresh=127,
    maxval=255,
    type=cv2.THRESH_OTSU
)

print("Umbral calculado por Otsu:", T_otsu)
print("Valores presentes después del umbralado:", np.unique(img_th))

# --- Visualización del umbralado ----------------------------------

plt.figure()

ax1 = plt.subplot(121)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Imagen original")
plt.xticks([])
plt.yticks([])

plt.subplot(122, sharex=ax1, sharey=ax1)
plt.imshow(img_th, cmap="gray", vmin=0, vmax=255)
plt.title(f"Umbralado - Otsu T={T_otsu:.0f}")
plt.xticks([])
plt.yticks([])

guardar_figura("p2_umbralado_otsu")
plt.show(block=True)

# --- Imagen binaria para análisis --------------------------------

img_bin = np.uint8(img_th == 0)

print("Valores de img_bin:", np.unique(img_bin))


suma_filas = np.sum(img_bin, axis=1)

plt.figure()
plt.plot(suma_filas)
plt.title("Cantidad de píxeles oscuros por fila")
plt.xlabel("Fila")
plt.ylabel("Cantidad de píxeles oscuros")
guardar_figura("p2_pixeles_oscuros_por_fila")
plt.show(block=True)


suma_columnas = np.sum(img_bin, axis=0)

plt.figure()
plt.plot(suma_columnas)
plt.title("Cantidad de píxeles oscuros por columna")
plt.xlabel("Columna")
plt.ylabel("Cantidad de píxeles oscuros")
guardar_figura("p2_pixeles_oscuros_por_columna")
plt.show(block=True)

# --- Detección de líneas ------------------------------------------

umbral_filas = 0.90 * np.max(suma_filas) #antes era 0.70 pero detectaba la linea del encabezado donde esta el nombre y la fecha
umbral_columnas = 0.70 * np.max(suma_columnas)

print("Umbral filas:", umbral_filas)
print("Umbral columnas:", umbral_columnas)

filas_con_linea = suma_filas > umbral_filas
columnas_con_linea = suma_columnas > umbral_columnas

idx_filas = np.argwhere(filas_con_linea).flatten()
idx_columnas = np.argwhere(columnas_con_linea).flatten()

print("Filas detectadas:")
print(idx_filas)

print("Columnas detectadas:")
print(idx_columnas)

# --- Inicio y fin de las líneas -----------------------------------

cambios_filas = np.diff(filas_con_linea) #Eso representa los lugares donde el vector cambia entre False y True
lineas_filas = np.argwhere(cambios_filas).flatten()

print("Cambios en filas:", lineas_filas)

lineas_filas[::2] += 1

lineas_filas = lineas_filas.reshape(-1, 2)

print("Inicio y fin de cada línea horizontal:")
print(lineas_filas)

# --- Inicio y fin de las líneas verticales ------------------------

cambios_columnas = np.diff(columnas_con_linea)
lineas_columnas = np.argwhere(cambios_columnas).flatten()

print("Cambios en columnas:", lineas_columnas)

# Corrijo los índices de inicio
lineas_columnas[::2] += 1

# Agrupo inicio y fin de cada línea
lineas_columnas = lineas_columnas.reshape(-1, 2)

print("Inicio y fin de cada línea vertical:")
print(lineas_columnas)

# --- Recorte automático de las preguntas -------------------------

preguntas_izq = []
preguntas_der = []

for i in range(5):

    fila_inicio = lineas_filas[i, 1] + 1
    fila_fin = lineas_filas[i + 1, 0]

    # Bloque izquierdo: preguntas 1 a 5
    col_inicio = lineas_columnas[0, 1] + 1
    col_fin = lineas_columnas[1, 0]

    pregunta = img[
        fila_inicio:fila_fin,
        col_inicio:col_fin
    ]

    preguntas_izq.append(pregunta)

    # Bloque derecho: preguntas 6 a 10
    col_inicio = lineas_columnas[2, 1] + 1
    col_fin = lineas_columnas[3, 0]

    pregunta = img[
        fila_inicio:fila_fin,
        col_inicio:col_fin
    ]

    preguntas_der.append(pregunta)

preguntas = preguntas_izq + preguntas_der

plt.figure()

for i in range(10):

    plt.subplot(5, 2, i + 1)
    plt.imshow(preguntas[i], cmap="gray", vmin=0, vmax=255)
    plt.title(f"Pregunta {i + 1}")
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
guardar_figura("p2_preguntas")
plt.show(block=True)

# --- Función para extraer las respuestas de una pregunta ----------

def extraer_componentes_respuesta(pregunta):

    # Umbralado
    _, pregunta_th = cv2.threshold(
        pregunta,
        thresh=127,
        maxval=255,
        type=cv2.THRESH_OTSU
    )

    # Fondo = 0, contenido = 1
    pregunta_bin = np.uint8(pregunta_th == 0)

    # Componentes conectadas de toda la pregunta
    n_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(
        pregunta_bin,
        8,
        cv2.CV_32S
    )

    # Busco candidatos a ser la línea de respuesta
    candidatos = []

    for i in range(1, n_comp):

        ancho = stats[i, cv2.CC_STAT_WIDTH]
        alto = stats[i, cv2.CC_STAT_HEIGHT]

        if ancho > 50 and ancho < pregunta_bin.shape[1] * 0.8 and alto <= 2:
            candidatos.append(i)

    # Si no encuentro línea, devuelvo vacío
    if len(candidatos) == 0:
        return [], None

    # Tomo la línea encontrada
    i_linea = candidatos[0]

    x_linea = stats[i_linea, cv2.CC_STAT_LEFT]
    y_linea = stats[i_linea, cv2.CC_STAT_TOP]
    ancho_linea = stats[i_linea, cv2.CC_STAT_WIDTH]
    alto_linea = stats[i_linea, cv2.CC_STAT_HEIGHT]

    # Borro solamente la línea de respuesta
    pregunta_sin_linea = pregunta_bin.copy()
    pregunta_sin_linea[labels == i_linea] = 0

    # Recorto la zona donde puede estar escrita la respuesta
    margen = 15

    zona_respuesta = pregunta_sin_linea[
        max(0, y_linea - margen):y_linea + alto_linea + 2,
        x_linea:x_linea + ancho_linea
    ]

    # Componentes conectadas dentro de la zona de respuesta
    n_resp, labels_resp, stats_resp, centroids_resp = cv2.connectedComponentsWithStats(
        zona_respuesta,
        8,
        cv2.CV_32S
    )

    componentes = []

    for i in range(1, n_resp):

        x = stats_resp[i, cv2.CC_STAT_LEFT]
        y = stats_resp[i, cv2.CC_STAT_TOP]
        ancho = stats_resp[i, cv2.CC_STAT_WIDTH]
        alto = stats_resp[i, cv2.CC_STAT_HEIGHT]
        area = stats_resp[i, cv2.CC_STAT_AREA]

        if area > 10:

            letra = np.uint8(
        labels_resp[y:y+alto, x:x+ancho] == i
        )
            componentes.append(
            (x, y, ancho, alto, area, letra)
        )

    return componentes, zona_respuesta


# --- Analizo las 10 preguntas -------------------------------------

for nro in range(10):

    componentes, zona = extraer_componentes_respuesta(
        preguntas[nro]
    )


    print("\nPregunta", nro + 1)

    print("Cantidad de componentes de respuesta:", len(componentes))

    for j in range(len(componentes)):

        x, y, ancho, alto, area, letra = componentes[j]

        print(
            "Componente", j + 1,
            "| ancho:", ancho,
            "| alto:", alto,
            "| area:", area
        )

# --- Prueba: mostrar la letra aislada de la pregunta 1 ------------

componentes, zona = extraer_componentes_respuesta(preguntas[0])

for i in range(len(componentes)):

    letra = componentes[i][5]

    plt.figure()
    plt.imshow(letra, cmap="gray")
    plt.title(f"Letra aislada {i + 1}")
    plt.xticks([])
    plt.yticks([])
    guardar_figura(f"p2_letra_aislada_{i + 1}")
    plt.show(block=True)



# --- Contar huecos de una letra -----------------------------------

def contar_huecos(letra):

    # Agrego un borde negro alrededor de la letra
    letra_borde = cv2.copyMakeBorder(
        letra,
        1, 1, 1, 1,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # Invierto: fondo y huecos pasan a valer 1
    fondo_letra = np.uint8(letra_borde == 0)

    # Busco componentes conectadas del fondo
    n_fondo, labels_fondo, stats_fondo, centroids_fondo = \
    cv2.connectedComponentsWithStats(
        fondo_letra,
        connectivity=4,
        ltype=cv2.CV_32S
    )

    # Componentes que tocan el borde = fondo exterior
    etiquetas_borde = np.unique(
        np.concatenate([
            labels_fondo[0, :],
            labels_fondo[-1, :],
            labels_fondo[:, 0],
            labels_fondo[:, -1]
        ])
    )

    huecos = 0

    for i in range(1, n_fondo):

        if i not in etiquetas_borde:
            huecos += 1

    return huecos


# --- Reconocer letra ----------------------------------------------

def reconocer_letra(componente):

    x, y, ancho, alto,      area, letra = componente

    huecos = contar_huecos(letra)

    if huecos == 2:
        return "B"

    elif huecos == 0:
        return "C"

    elif huecos == 1:

        # A es más ancha que D
        if ancho >= 8:
            return "A"
        else:
            return "D"

    return "Desconocida"

# --- Respuestas correctas -----------------------------------------

respuestas_correctas = [
    "C", "B", "A", "D", "B",
    "B", "A", "B", "D", "D"
]

# --- Corrección de las respuestas --------------------------------

aciertos = 0

for nro in range(10):

    componentes, zona = extraer_componentes_respuesta(
        preguntas[nro]
    )

    print("\nPregunta", nro + 1)

    if len(componentes) == 0:

        print("Sin respuesta")
        print("Resultado: MAL")

    elif len(componentes) > 1:

        print("Más de una respuesta")
        print("Resultado: MAL")

    else:

        respuesta = reconocer_letra(componentes[0])
        correcta = respuestas_correctas[nro]

        print("Respuesta detectada:", respuesta)
        print("Respuesta correcta:", correcta)

        if respuesta == correcta:
            print("Resultado: OK")
            aciertos += 1
        else:
            print("Resultado: MAL")

print("\nCantidad de respuestas correctas:", aciertos)

if aciertos >= 6:
    print("Resultado final: APROBADO")
else:
    print("Resultado final: DESAPROBADO")










# --- Recorte del encabezado ---------------------------------------

fila_fin_encabezado = lineas_filas[0, 0]

encabezado = img[
    0:fila_fin_encabezado,
    :
]

plt.figure()
plt.imshow(encabezado, cmap="gray", vmin=0, vmax=255)
plt.title("Encabezado")
plt.xticks([])
plt.yticks([])
guardar_figura("p2_encabezado")
plt.show(block=True)


# --- Detección de los campos del encabezado ----------------------

_, encabezado_th = cv2.threshold(
    encabezado,
    thresh=127,
    maxval=255,
    type=cv2.THRESH_OTSU
)

encabezado_bin = np.uint8(encabezado_th == 0)

n_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(
    encabezado_bin,
    8,
    cv2.CV_32S
)

lineas_campos = []

for i in range(1, n_comp):

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    ancho = stats[i, cv2.CC_STAT_WIDTH]
    alto = stats[i, cv2.CC_STAT_HEIGHT]

    if ancho > 50 and alto <= 2:

        lineas_campos.append(
            (i, x, y, ancho, alto)
        )

        print(
            "Candidato",
            "| x:", x,
            "| y:", y,
            "| ancho:", ancho,
            "| alto:", alto
        )

lineas_campos.sort(key=lambda campo: campo[1])


encabezado_sin_lineas = encabezado_bin.copy()

for campo in lineas_campos:

    i_linea = campo[0]

    encabezado_sin_lineas[
        labels == i_linea
    ] = 0


campos = []

margen = 18

for campo in lineas_campos:

    i_linea, x, y, ancho, alto = campo

    recorte = encabezado_sin_lineas[
        max(0, y - margen):y + alto + 2,
        x:x + ancho
    ]

    campos.append(recorte)


nombres_campos = ["Name", "Date", "Class"]

plt.figure()

for i in range(3):

    plt.subplot(1, 3, i + 1)
    plt.imshow(campos[i], cmap="gray")
    plt.title(nombres_campos[i])
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
guardar_figura("p2_campos_encabezado")
plt.show(block=True)

# --- Análisis de componentes de los campos -----------------------

for nro in range(3):

    campo = campos[nro]

    n_comp, labels_campo, stats_campo, centroids_campo = \
        cv2.connectedComponentsWithStats(
            campo,
            8,
            cv2.CV_32S
        )

    print("\nCampo:", nombres_campos[nro])

    for i in range(1, n_comp):

        x = stats_campo[i, cv2.CC_STAT_LEFT]
        y = stats_campo[i, cv2.CC_STAT_TOP]
        ancho = stats_campo[i, cv2.CC_STAT_WIDTH]
        alto = stats_campo[i, cv2.CC_STAT_HEIGHT]
        area = stats_campo[i, cv2.CC_STAT_AREA]

        print(
            "Componente", i,
            "| x:", x,
            "| ancho:", ancho,
            "| alto:", alto,
            "| area:", area
        )

# --- Distancias entre caracteres del Name ------------------------

campo_name = campos[0]

n_comp, labels_name, stats_name, centroids_name = \
    cv2.connectedComponentsWithStats(
        campo_name,
        8,
        cv2.CV_32S
    )

for i in range(1, n_comp - 1):

    x_actual = stats_name[i, cv2.CC_STAT_LEFT]
    ancho_actual = stats_name[i, cv2.CC_STAT_WIDTH]

    x_siguiente = stats_name[i + 1, cv2.CC_STAT_LEFT]

    fin_actual = x_actual + ancho_actual

    espacio = x_siguiente - fin_actual

    print(
        "Entre componente", i,
        "y", i + 1,
        "| espacio:", espacio
    )

# --- Validamos Name ------------------------

def validar_name(campo_name):

    n_comp, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            campo_name,
            8,
            cv2.CV_32S
        )

    cantidad_caracteres = n_comp - 1
    cantidad_espacios = 0

    for i in range(1, n_comp - 1):

        x_actual = stats[i, cv2.CC_STAT_LEFT]
        ancho_actual = stats[i, cv2.CC_STAT_WIDTH]

        x_siguiente = stats[i + 1, cv2.CC_STAT_LEFT]

        fin_actual = x_actual + ancho_actual
        espacio = x_siguiente - fin_actual

        if espacio > 3:
            cantidad_espacios += 1

    cantidad_palabras = cantidad_espacios + 1

    total_caracteres = cantidad_caracteres + cantidad_espacios

    if cantidad_palabras >= 2 and total_caracteres <= 25:
        return "OK"
    else:
        return "MAL"

print("Name:", validar_name(campos[0]))


# --- Validamos Date ---------------------------------------------

def validar_date(campo_date):

    n_comp, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            campo_date,
            8,
            cv2.CV_32S
        )

    componentes = []

    # Filtro ruido
    for i in range(1, n_comp):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > 10:
            componentes.append(i)

    # Debe tener exactamente 8 caracteres
    if len(componentes) != 8:
        return "MAL"

    # Ordeno los caracteres de izquierda a derecha
    componentes.sort(
        key=lambda i: stats[i, cv2.CC_STAT_LEFT]
    )

    # Compruebo que formen una sola palabra
    for j in range(len(componentes) - 1):

        i_actual = componentes[j]
        i_siguiente = componentes[j + 1]

        x_actual = stats[i_actual, cv2.CC_STAT_LEFT]
        ancho_actual = stats[i_actual, cv2.CC_STAT_WIDTH]

        x_siguiente = stats[i_siguiente, cv2.CC_STAT_LEFT]

        fin_actual = x_actual + ancho_actual
        espacio = x_siguiente - fin_actual

        if espacio > 5:
            return "MAL"

    return "OK"

print("Date:", validar_date(campos[1]))


# --- Validamos Class ---------------------------------------------

def validar_class(campo_class):

    n_comp, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            campo_class,
            8,
            cv2.CV_32S
        )

    componentes = []

    for i in range(1, n_comp):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > 10:
            componentes.append(i)

    if len(componentes) == 1:
        return "OK"
    else:
        return "MAL"


print("Class:", validar_class(campos[2]))

# --- Resultado del encabezado -----------------------------------

estado_name = validar_name(campos[0])
estado_date = validar_date(campos[1])
estado_class = validar_class(campos[2])

print("\n--- Encabezado ---")
print("Name:", estado_name)
print("Date:", estado_date)
print("Class:", estado_class)











def procesar_examen(ruta):

    img = cv2.imread(
        ruta,
        cv2.IMREAD_GRAYSCALE
    )

    # --- Umbralado -------------------------------------

    T_otsu, img_th = cv2.threshold(
        img,
        thresh=127,
        maxval=255,
        type=cv2.THRESH_OTSU
    )

    img_bin = np.uint8(img_th == 0)

    # --- Detectar líneas de la tabla -------------------

    suma_filas = np.sum(img_bin, axis=1)
    suma_columnas = np.sum(img_bin, axis=0)

    umbral_filas = 0.90 * np.max(suma_filas)
    umbral_columnas = 0.70 * np.max(suma_columnas)

    filas_con_linea = suma_filas > umbral_filas
    columnas_con_linea = suma_columnas > umbral_columnas

    cambios_filas = np.diff(filas_con_linea)
    lineas_filas = np.argwhere(cambios_filas).flatten()
    lineas_filas[::2] += 1
    lineas_filas = lineas_filas.reshape(-1, 2)

    cambios_columnas = np.diff(columnas_con_linea)
    lineas_columnas = np.argwhere(cambios_columnas).flatten()
    lineas_columnas[::2] += 1
    lineas_columnas = lineas_columnas.reshape(-1, 2)

    # --- Recortar preguntas ----------------------------

    preguntas_izq = []
    preguntas_der = []

    for i in range(5):

        fila_inicio = lineas_filas[i, 1] + 1
        fila_fin = lineas_filas[i + 1, 0]

        # Izquierda
        col_inicio = lineas_columnas[0, 1] + 1
        col_fin = lineas_columnas[1, 0]

        pregunta = img[
            fila_inicio:fila_fin,
            col_inicio:col_fin
        ]

        preguntas_izq.append(pregunta)

        # Derecha
        col_inicio = lineas_columnas[2, 1] + 1
        col_fin = lineas_columnas[3, 0]

        pregunta = img[
            fila_inicio:fila_fin,
            col_inicio:col_fin
        ]

        preguntas_der.append(pregunta)

    preguntas = preguntas_izq + preguntas_der

        # --- Corrección de las respuestas ----------------------------

    respuestas_correctas = [
        "C", "B", "A", "D", "B",
        "B", "A", "B", "D", "D"
    ]

    aciertos = 0

    for nro in range(10):

        componentes, zona = extraer_componentes_respuesta(
            preguntas[nro]
        )

        if len(componentes) == 1:

            respuesta = reconocer_letra(componentes[0])
            correcta = respuestas_correctas[nro]

            if respuesta == correcta:
                aciertos += 1

    if aciertos >= 6:
        resultado = "APROBADO"
    else:
        resultado = "DESAPROBADO"

        # --- Procesamiento del encabezado ----------------------------

    fila_fin_encabezado = lineas_filas[0, 0]

    encabezado = img[
        0:fila_fin_encabezado,
        :
    ]

    _, encabezado_th = cv2.threshold(
        encabezado,
        thresh=127,
        maxval=255,
        type=cv2.THRESH_OTSU
    )

    encabezado_bin = np.uint8(encabezado_th == 0)

    n_comp, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            encabezado_bin,
            8,
            cv2.CV_32S
        )

    # Busco las tres líneas: Name, Date y Class
    lineas_campos = []

    for i in range(1, n_comp):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        ancho = stats[i, cv2.CC_STAT_WIDTH]
        alto = stats[i, cv2.CC_STAT_HEIGHT]

        if ancho > 50 and alto <= 2:
            lineas_campos.append(
                (i, x, y, ancho, alto)
            )

    # Las ordeno de izquierda a derecha
    lineas_campos.sort(
        key=lambda campo: campo[1]
    )

    # Borro las líneas
    encabezado_sin_lineas = encabezado_bin.copy()

    for campo in lineas_campos:

        i_linea = campo[0]

        encabezado_sin_lineas[
            labels == i_linea
        ] = 0

    # Recorto los tres campos
    campos = []
    margen = 18

    for campo in lineas_campos:

        i_linea, x, y, ancho, alto = campo

        recorte = encabezado_sin_lineas[
            max(0, y - margen):y + alto + 2,
            x:x + ancho
        ]

        campos.append(recorte)

    # --- Validación -----------------------------------------------

    estado_name = validar_name(campos[0])
    estado_date = validar_date(campos[1])
    estado_class = validar_class(campos[2])

    return (
    aciertos,
    resultado,
    estado_name,
    estado_date,
    estado_class,
    campos[0]
)


# --- Procesamiento de todos los exámenes -------------------------

resultados_examenes = []

nombres_aprobados = []
nombres_desaprobados = []

for nro in range(1, 6):

    ruta = f"Imagenes/examen_{nro}.png"

    aciertos, resultado, estado_name, estado_date, estado_class, nombre = \
        procesar_examen(ruta)

    resultados_examenes.append(
        (
            nro,
            aciertos,
            resultado,
            estado_name,
            estado_date,
            estado_class
        )
    )

    if resultado == "APROBADO":
        nombres_aprobados.append(nombre)
    else:
        nombres_desaprobados.append(nombre)

    print("\n--- Examen", nro, "---")
    print("Aciertos:", aciertos)
    print("Resultado:", resultado)
    print("Name:", estado_name)
    print("Date:", estado_date)
    print("Class:", estado_class)


# --- Imagen final de resultados ---------------------------------

# Colores según condición (verde = aprobado, rojo = no aprobado)
COLOR_APROBADO = (0.0, 0.55, 0.0)
COLOR_DESAPROBADO = (0.75, 0.0, 0.0)


def colorear_nombre(nombre_bin, color):
    """
    Convierte el recorte binario del campo Name (0 = fondo,
    1 = tinta) en una imagen RGB con fondo blanco y el texto
    pintado del color indicado.
    """
    alto, ancho = nombre_bin.shape
    imagen_rgb = np.ones((alto, ancho, 3))
    imagen_rgb[nombre_bin == 1] = color

    return imagen_rgb


# Uno los nombres aprobados y desaprobados en un solo listado,
# guardando junto a cada uno el color y la etiqueta que le
# corresponden.
todos_los_nombres = []

for nombre in nombres_aprobados:
    todos_los_nombres.append((nombre, COLOR_APROBADO, "APROBADO"))

for nombre in nombres_desaprobados:
    todos_los_nombres.append((nombre, COLOR_DESAPROBADO, "DESAPROBADO"))

n_alumnos = len(todos_los_nombres)

fig, axes = plt.subplots(
    n_alumnos, 1,
    figsize=(6, 1.3 * n_alumnos + 1)
)

if n_alumnos == 1:
    axes = [axes]

fig.suptitle(
    "Verde: Aprobado   |   Rojo: No aprobado",
    fontsize=14,
    fontweight="bold"
)

for idx, (nombre, color, resultado) in enumerate(todos_los_nombres):

    imagen_nombre = colorear_nombre(nombre, color)

    ax = axes[idx]
    ax.imshow(imagen_nombre)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        resultado,
        fontsize=9,
        color=color
    )

plt.tight_layout(rect=[0, 0, 1, 0.94])
guardar_figura("p2_resumen_alumnos")
plt.show(block=True)