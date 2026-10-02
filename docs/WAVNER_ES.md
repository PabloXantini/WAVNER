# Funcionamiento de WAVNER

WAVNER funciona a partir del reconomiento de gestos con las manos en tiempo real mediante Mediapipe, principalmente usando 3 canales de audio, donde se puede personalizar para establecer cualquier tipo de señal simple.

> _**Nota:** En esta versión del software esta en desarrollo y por el momento la personalización solo se aplica modificando los archivos de este repositorio._

## Descripción de los canales de audio

* **Canales de fondo:** Son los canales donde se reproduce música o sonidos de fondo, estos tienen modulación limitada con el gesto de la mano.

* **Canal principal:** Se destina para la modulación mediante gestos con modificadores los cuales son:
    * Filtros altos (High Pass)
    * Filtros bajos (Low Pass)
    * Reverberación/Eco (Reverb)
    * Distorsión (Distortion)
    * Volumen
    * Tonalidad (Pitch)
* **Canal extra:** Es un canal adicional donde se puede intercambiar señales adicionales con las teclas de un teclado.

## Controles

### Mano Izquierda
* **Gesto de dedo meñique (doblez):** Volumen del Canal 1 de fondo.
* **Gesto de dedos anular-medio (doblez):** Volumen del Canal 2 de fondo.
* **Gesto de dedo índice (doblez):** Modificador de distorción para el Canal principal.
* **Gesto de dedo pulgar (doblez):** Modificador de reverberación para el Canal principal.

## Mano Derecha
* **Gesto de dedo pulgar-índice (pincheo):** Modificar del filtro alto para el Canal principal
* **Gesto de dedo medio/anular-meñique (pincheo):** Modificar el filtro bajo para el Canal principal

# Teclado
<table style="margin: auto;">
    <thead>
        <th>Tecla</th>
        <th>Acción</th>
    </thead>
    <tbody>
        <tr>
            <td>1</td>
            <td>Establecer la señal <b>sinoidal</b> al Canal Extra.</td>
        </tr>
        <tr>
            <td>2</td>
            <td>Establecer la señal <b>cuadrada</b> al Canal Extra.</td>
        </tr>
        <tr>
            <td>3</td>
            <td>Establecer la señal <b>triangular</b> al Canal Extra.</td>
        </tr>
        <tr>
            <td>4</td>
            <td>Establecer la señal <b>diente de sierra</b> al Canal Extra.</td>
        </tr>
        <tr>
            <td>ENTER</td>
            <td>Guardar el Canal Extra.</td>
        </tr>
        <tr>
            <td>ESPACIO</td>
            <td>Desactivar el Canal Extra.</td>
        </tr>
    </tbody>
</table>
