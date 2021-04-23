import cv2
import math

"""
    Utilizando el algoritmo Meanshift para detectar movimiento.
"""
# Se accede a la camara
camara = cv2.VideoCapture(2)

# Se lee una imagen para saber las dimensiones de las imágenes
_, imagen = camara.read()

# Primera área sobre la que se va a aplicar Meanshift
wCamara = int(camara.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
hCamara = int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
x, y, w, h = wCamara, hCamara, 100, 100
ventana = (x, y, w, h)

# Se crea un área de interés
areaInteres = imagen[y:y + h, x:x + w]

# Pasamos la imágen de RGB a HSV
hsv = cv2.cvtColor(areaInteres, cv2.COLOR_BGR2HSV)

# Creamos un histograma para HSV
histograma = cv2.calcHist([hsv], [0], None, [180], [0, 180])

# Se normalizan los valores
histograma = cv2.normalize(histograma, histograma, 0, 255, cv2.NORM_MINMAX)

# Punto sobre el que calcular la distancia
x0 = -1
y0 = -1

# Umbral que determina si existe movimiento
umbral = 0.04 * math.sqrt(camara.get(cv2.CAP_PROP_FRAME_HEIGHT) ** 2 + camara.get(cv2.CAP_PROP_FRAME_WIDTH) ** 2)

"""
    Para que MeanShift termine se pude fijar un número de iteraciones máximas o
    marcar una condición de convergencia. En opencv se pueden utilizar los dos
    criterios a la vez o tan solo uno.

    * El primer parámetro es que método se va a utilizar.

    * Cantidad de iteraciones máximas

    * Cantidad de píxeles a los que se desplaza. Si es menor se detiene.
"""
terminar = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 1)

while True:
    # Se consume una imagen
    _, imagen = camara.read()

    # Pasamos la imágen de RGB a HSV
    imagenHSV = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Se aplica el histograma de HSV creado anteriormente y se crea una mascara
    mascara = cv2.calcBackProject([imagenHSV], [0], histograma, [0, 180], 1)

    # Se aplica Meanshift
    _, ventana = cv2.meanShift(mascara, ventana, terminar)

    # Obtenemos la información de la ventana
    x, y, w, h = ventana

    # calculamos la distancia entre el punto anterior y el actual
    if x0 == -1 or y0 == -1:
        x0 = x
        y0 = y
    else:
        # Se calcula la distancia entre dos puntos
        distancia = math.sqrt( (x0 - x) ** 2 + (y0 - y) **2)

        # Se renuevan los valores de los puntos
        x0 = x
        y0 = y

        # Si la distancia supera cierto umbral se detecta movimiento
        if distancia > umbral:
            print("movimiento")

    # Se dibuja la ventana en el frame
    x, y, w, h = ventana
    imagenSalida = cv2.rectangle(imagen, (x, y), (x + w, y + h), 255, 2)

    # Se muestra la imagen con la máscara aplicada sobre el histograma
    cv2.imshow("Meanshift - Mascara", mascara)

    # Se muestra la imagen con el cuadrado
    cv2.imshow("Meanshift - HSV", imagenSalida)

    # Pulsar 'q' para salir
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break