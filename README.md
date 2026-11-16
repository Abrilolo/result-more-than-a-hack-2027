# Pitch Controllers — posesión H/A/D, BAR-ATH (More Than A Hack 2027)

Pipeline replicable que, a partir del tracking automático oficial del reto (ATD), produce
`prediction/Pitch_Controllers.csv` (199 319 filas, `frame,label`). Documento técnico:
[Pitch_Controllers.md](Pitch_Controllers.md).

## Estructura

| ruta | qué es |
|---|---|
| `run.py` | único punto de entrada (`predict`, `train`, `validate`, `public`) |
| `src/config.py` | **todas** las decisiones fijas del método (semilla, modelo, decodificador) |
| `src/target_inputs.py` | lectura del ATD oficial (`.txt` + `.xml`) a metros, con el equipo de cada slot |
| `src/idsse.py` | descarga/parsing de los 7 partidos públicos IDSSE (DFL, CC BY 4.0) a 25 Hz |
| `src/labels.py` | regla de *posesión controlada* aplicada a las etiquetas DFL |
| `src/domain.py` | degradación de IDSSE para que se parezca al tracking automático del reto |
| `src/features.py` | extractor de características común (IDSSE y reto) |
| `src/decode.py` | decodificador HSMM (duración mínima + penalizaciones de cambio) |
| `src/restarts.py` | reglas de balón parado: saque inicial de cada parte y balón parado en el área |
| `src/pipeline.py` | entrenamiento, validación leave-one-match-out, inferencia |
| `src/possession_metrics.py` | copia literal del evaluador oficial |
| `model/` | **modelo entrenado** (`model.ubj`) y `MODEL.json` (proporción de clases, columnas, hash) |
| `external/idsse/` | **los 7 partidos IDSSE ya procesados** (25 Hz, metros, etiquetas DFL) con los que se entrena |
| `data/tactical/` | datos del reto (privados, en `.gitignore`, nunca se publican) |
| `reference/` | GT público de 5 min (solo comprobación de formato/alineación) |
| `prediction/` | CSV final + `MANIFEST.json` (hash del CSV y del modelo, configuración) |
| `reports/` | validación IDSSE y chequeo público (informe completo del evaluador oficial) |

`cache/` (ATD en metros y tablas de características) se regenera solo. `external/raw/` solo hace
falta si se borra `external/idsse/`: entonces los XML de IDSSE se descargan desde una revisión
fijada y se vuelven a procesar.

## Datos necesarios

Del paquete de datos oficial del reto se usan tres archivos, que deben estar en
`data/tactical/` (o en `$MTAH_DATA_DIR/tactical/`):

```text
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

Del Smart Tagging solo se toma la hora de la etiqueta "Kick Off (Start)", como ancla para detectar
el saque inicial de cada parte. No se usan el vídeo, la homografía ni el broadcast.

**Los datos del reto no están en este repositorio, a propósito:** el reglamento prohíbe
redistribuirlos o publicar derivados que los reconstruyan, así que `.gitignore` excluye `data/` y
`cache/`. Quien revise el proyecto ya tiene el paquete oficial y solo tiene que copiar esos tres
archivos. Lo que sí se publica es todo lo que generamos a partir de datos públicos: el modelo
entrenado y los partidos IDSSE procesados. El detalle de qué dato se usa, para qué y dónde vive
está en [Pitch_Controllers.md](Pitch_Controllers.md) (§2).

## Reproducir (Windows, PowerShell)

1. Copiar los tres archivos del paquete de datos oficial en `data/tactical/` (la carpeta no viene
   en el repositorio; hay que crearla). Ajustar `$datos` a donde esté descomprimido el paquete:

```powershell
$datos = 'C:\ruta\al\paquete\data'
New-Item -ItemType Directory -Force data\tactical | Out-Null
Copy-Item "$datos\tactical\BAR-ATH_tactical_atd.txt", "$datos\tactical\BAR-ATH_tactical_atd.xml", "$datos\tactical\BAR-ATH_tactical_smart_tagging.xml" data\tactical\
```

   También se puede dejar el paquete donde está y apuntar a él: `$env:MTAH_DATA_DIR = $datos`.
   Si falta alguno de los tres archivos, `run.py` se detiene con un mensaje que dice cuál y dónde
   copiarlo; nunca genera un CSV sin ellos.

2. Instalar el entorno y generar el CSV:

```powershell
& .\setup.ps1
& .\.venv\Scripts\python.exe .\run.py predict    # CSV con el modelo publicado en model/ (~30 s)
& .\.venv\Scripts\python.exe .\run.py public     # chequeo con el GT público: todas las métricas oficiales
```

Para comprobar también el entrenamiento:

```powershell
& .\.venv\Scripts\python.exe .\run.py train      # reentrena con external/idsse/ y reescribe model/ (~8 min)
& .\.venv\Scripts\python.exe .\run.py predict    # mismo CSV byte a byte
& .\.venv\Scripts\python.exe .\run.py validate   # evidencia de selección: IDSSE leave-one-match-out (~35 min)
```

Todo es determinista: `train` produce el mismo `model.ubj` (mismo hash que `model/MODEL.json`) y
`predict` el mismo CSV (hash en `prediction/MANIFEST.json`). No hace falta internet salvo para
instalar los paquetes.
