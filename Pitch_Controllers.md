# Pitch Controllers — documento técnico: posesión H/A/D a partir del tracking automático

## 1. El enfoque en un párrafo

Un clasificador de gradient boosting (XGBoost) estima P(H), P(A) y P(D) en cada frame, a 25 Hz,
a partir de la geometría de los 22 jugadores y del balón. Después, un decodificador semi-Markov
oculto convierte esas probabilidades en posesiones con una duración mínima y penalizaciones por
cambio. El clasificador se entrena **solo con datos externos**: los 7 partidos públicos de la
Bundesliga de IDSSE (DFL / Bassek et al. 2025, CC BY 4.0). Su tracking limpio se **degrada
para que se parezca al tracking automático del reto**: el patrón exacto de presencia/ausencia
del balón del reto, deriva suave de las posiciones y jugadores que desaparecen. Todas las
decisiones se tomaron con el **spellF1 leave-one-match-out sobre IDSSE**, nunca con el partido
del reto. Los 5 minutos públicos solo se usaron para comprobar que frames, etiquetas y formato
del archivo son correctos.

## 2. Datos usados, probados y descartados

| dato | uso |
|---|---|
| ATD táctico oficial (`BAR-ATH_tactical_atd.txt` + `.xml`: jugadores, equipo de cada slot, balón) | **usado** en todas las características; `src/target_inputs.py` lo pasa de coordenadas 0–1 a metros (105.3 × 68). El balón está presente en el 59 % de los frames evaluados |
| Homografía táctica | no usada (el ATD ya está en coordenadas del campo) |
| Salida del detector/tracker propio (`half_*.sqlite`) | **probada y descartada** para el balón: con score ≥ 0.3 solo el 57 % de las detecciones cae a menos de 2 m del balón ATD. Un subconjunto con soporte temporal y cercano a jugadores rellena solo el 13 % de los huecos del balón ATD y se equivoca (> 2 m) el 23 % de las veces. Mezclarlo mete valores atípicos para los que el modelo no fue entrenado. No forma parte de la entrega |
| Pipeline anterior (jugadores re-proyectados desde detecciones propias, 5 Hz, 29 características, HMM) | sustituido. Tomaba la caja de balón con mayor score de cada frame sin asociación temporal, re-proyectaba jugadores en lugar de usar el ATD y decodificaba a 5 Hz |
| Smart Tagging | no usado: no tiene equivalente en los datos de entrenamiento, así que su peso no se podía validar |
| Vídeo y tracking de broadcast | no usados |

## 3. Datos de entrenamiento

* IDSSE — 7 partidos (J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9), archivos de
  posiciones DFL en la revisión fijada `a715a38d` de `pysport/idsse-data`. Etiquetas:
  `BallStatus` de DFL (balón muerto → D) y `BallPossession` (local/visitante) a 25 Hz,
  producidas por el proveedor de datos de DFL, no por nosotros.
* No se usa ninguna etiqueta del partido del reto, de ningún tipo, ni para entrenar, ni para
  seleccionar, ni para decodificar.

### Convención de etiquetas: posesión controlada
La posesión de DFL cambia con cada toque (mediana de 3–4 s por posesión). El reto solo cambia la
posesión cuando el rival gana el **control** ("un simple toque sin control", los desvíos y los
duelos aéreos no cuentan; "un jugador conserva el balón unos segundos"). `src/labels.py` aplica
esa regla: una racha del equipo que no tiene la posesión que dure menos de **2 s** se asigna al
equipo que la tenía; el balón muerto no se toca; el primer equipo tras una reanudación siempre es
el poseedor. Los 2 s salen del texto del reglamento, no de un ajuste.

## 4. Características (`src/features.py`, idénticas para IDSSE y para el reto)

Expresadas en la dirección de ataque del equipo local (inferida en cada parte a partir de los
centroides de los equipos): posición del balón, distancia fuera del campo y velocidad; jugador
más cercano y segundo más cercano de cada equipo, y número de jugadores a menos de 2/5/10 m;
velocidad del jugador más cercano y su velocidad relativa al balón; un indicador de **control**
(jugador más cercano a menos de 1.5 m y balón moviéndose con él), con el último/próximo equipo
en control y el tiempo desde/hasta entonces (sirve para cruzar los huecos del balón); posición,
largo, ancho, velocidad, velocidad hacia delante y número de jugadores rápidos del bloque de cada
equipo; medias móviles centradas (1, 4 y 12 s); relojes de hueco del balón y estado del balón en
los dos extremos del hueco actual.

Simetría de equipos: cada fila de entrenamiento se duplica intercambiando local y visitante, y
en la inferencia se promedian la predicción directa y la intercambiada, así que no hay sesgo
hacia el local.

## 5. Adaptación de dominio (`src/domain.py`)

* Disponibilidad del balón: a cada parte de IDSSE se le aplica una ventana aleatoria de la
  **máscara real de disponibilidad del balón del reto**, con la misma longitud y frecuencia de
  huecos que habrá en la inferencia.
* Error de posición como deriva suave (balón σ 0.7 m, jugadores σ 0.4 m). Primero se probó ruido
  blanco por frame y era un error: multiplicaba por 2.5 las velocidades entre frames respecto al
  ATD del reto, y el modelo terminaba marcando casi todo como balón muerto.
* Jugadores que desaparecen (bloques de Markov encendido/apagado, 6 %), para igualar los ~10.2
  jugadores visibles por equipo.

## 6. Decodificador (`src/decode.py`)

Viterbi con duración explícita: cada etiqueta es una cadena de subestados que hay que recorrer
antes de salir (duraciones mínimas de posesión y de balón muerto), con una penalización por
cambio equipo↔equipo y otra vivo↔muerto, sobre las probabilidades divididas por la proporción de
clases del entrenamiento. Funciona en bloques de 5 frames; después cada frontera se mueve al
mejor frame dentro de ±5 frames usando la evidencia a 25 Hz. No hay abstención: en el spellF1 una
`X` solo repite la etiqueta anterior, y todos los frames evaluados tienen jugadores en el ATD.

## 7. Reglas de posesión implementadas

* Balones sueltos, entradas sin control, desvíos, duelos aéreos: la posesión se mantiene (regla
  de control de 2 s en las etiquetas + duración mínima de posesión en el decodificador).
* Reanudaciones: D hasta que se reanuda el juego; el equipo que reanuda es dueño de la primera
  racha viva.
* Nunca se predice a quién pertenece un balón muerto (siempre D).

## 8. Validación (evidencia de selección)

Protocolo: leave-one-match-out sobre los 7 partidos de IDSSE (se entrena con 6, se decodifica y
se puntúa el séptimo), con entradas degradadas como en §5, etiquetas con la regla de control de
§3 y la función oficial `spell_scores` de `possession_metrics.py`, juntando las dos partes de cada
partido y promediando entre partidos. Cada paso se conservó solo si mejoraba este número.

| paso | spellF1 LOMO IDSSE |
|---|---|
| primer modelo (degradación con ruido blanco, características v1, 400 árboles) | 0.633 |
| degradación con deriva suave (corrige el desajuste de velocidades) | 0.657 |
| + stacking temporal de segunda etapa | 0.658 — descartado (sin mejora, más complejidad) |
| + características de control (v2) | 0.664 |
| + modelo más grande (900 árboles, profundidad 8) | 0.675 |
| + 3 degradaciones independientes por parte (final) | **0.684** |

Los parámetros del decodificador se eligieron sobre las mismas predicciones fuera de pliegue
(posesión mínima 2 s, balón muerto mínimo 1 s, penalización por cambio de equipo 12,
penalización vivo/muerto 40); los valores vecinos quedan a ±0.002. La fusión con las detecciones
propias del balón se descartó antes de modelar (ver §2).

Chequeo de sanidad sobre los 5 minutos públicos, ejecutado una sola vez sobre el archivo final y
**no usado para ninguna decisión**: spellF1 0.629, macroF1 0.822, exactitud 0.859 (pipeline
anterior: 0.322 / 0.591 / 0.656). En el partido completo el archivo tiene 259 posesiones con una
media de 14.3 s.

## 9. Limitaciones

* Las etiquetas de entrenamiento son las de DFL, transformadas con nuestra regla de control; la
  definición de control del reto (de Metrica) puede ser distinta. Es el mayor riesgo y no se
  puede medir.
* Las posesiones cortas (< 6 s) se detectan con mucha menos fiabilidad que las largas.
* El balón falta en el 41 % de los frames evaluados; durante los huecos largos la decisión
  depende de la forma colectiva y del estado del balón en los extremos del hueco.
* Con más tiempo: un tracker de balón aprendido sobre las detecciones propias, y las
  reanudaciones del Smart Tagging como información previa validada.

## 10. Reproducción

Ver `README.md`: copiar `BAR-ATH_tactical_atd.txt` y `.xml` del paquete oficial en
`data/tactical/`, ejecutar `setup.ps1` y después `python run.py predict`. Es determinista; el hash
del CSV queda en `prediction/MANIFEST.json`.
