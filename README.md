<div align="center">

#  Procesamiento de Imágenes — Trabajo Práctico N°1

**Universidad Austral · Licenciatura en Ciencia de Datos · 2026**


</div>

---

## 👥 Integrantes

| Apellido y nombre |
|---|
| Castillo, Juan Ignacio |
| Fasolato, Matías |
| Torregiani, Bautista |

---

## 📌 Introducción

Este repositorio contiene la resolución del Trabajo Práctico N°1 de Procesamiento de Imágenes, compuesto por dos problemas:

| Problema | Descripción |
|---|---|
| **1 — Ecualización local de histograma** | Función que ecualiza el histograma de forma local con una ventana de M×N que recorre la imagen píxel a píxel. Se aplica sobre una imagen para revelar sus detalles ocultos y se analiza la influencia del tamaño de la ventana. |
| **2 — Corrección de multiple choice** | Script que, a partir solo de la imagen de un examen, corrige las 10 preguntas, valida los campos del encabezado (Name, Date y Class) e informa si el alumno aprobó (6 o más respuestas correctas). |

---

## 📁 Estructura del proyecto

```
Procesamiento-Imagenes/
├── Imagenes/
│   ├── Imagen_con_detalles_escondidos.tif   # Imagen del Problema 1
│   └── examen_1.png ... examen_5.png        # Exámenes del Problema 2
├── Imagenes-Resultados/
│   ├── p1_comparacion_ventanas.png          # Imagen de las ventanas en el Problema 1
│   └── p2_resumen_alumnos.png               # Resultado de los Exámenes del Problema 2
│   └── ...
├── problema1_ecualizacion_local.py          # Resolución del Problema 1
├── problema2_examen.py                      # Resolución del Problema 2
├── requirements.txt                         # Dependencias
└── README.md
```

---

## ⚙️ Instalación

**1. Clonar el repositorio**

```bash
git clone https://github.com/castillojuanignacio/Procesamiento-Imagenes.git
cd Procesamiento-Imagenes
```

**2. Crear y activar un entorno virtual**

```bash
python -m venv venv
```

| Sistema | Comando de activación |
|---|---|
| Windows | `venv\Scripts\activate` |
| Linux / macOS | `source venv/bin/activate` |

**3. Instalar las dependencias**

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecución

> [!IMPORTANTE]
> Los scripts deben ejecutarse **desde la carpeta raíz del repositorio**, ya que leen las imágenes de la carpeta `Imagenes/` con rutas relativas.
----

### 🔍 Problema 1 — Ecualización local de histograma

```bash
python problema1_ecualizacion_local.py
```

- **Entrada:** `Imagenes/Imagen_con_detalles_escondidos.tif`
- **Salida:** información básica de la imagen por consola y gráficos con la imagen original, su histograma y el resultado de la ecualización local con ventanas de **7×7, 15×15, 31×31 y 63×63**.
-----

### 📝 Problema 2 — Corrección de multiple choice

```bash
python problema2_examen.py
```

- **Entrada:** `Imagenes/examen_1.png` a `Imagenes/examen_5.png`
- **Salida:**
 
  - para los 5 exámenes, por consola: la cantidad de aciertos, el resultado final (`APROBADO` / `DESAPROBADO`) y la validación de los campos `Name`, `Date` y `Class` (`OK` / `MAL`).

**Respuestas correctas utilizadas:**

| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| C | B | A | D | B | B | A | B | D | D |
