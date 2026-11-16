# Pitch Controllers — posesión H/A/D, BAR-ATH (More Than A Hack 2027)

Pipeline replicable que, a partir del tracking automático oficial del reto (ATD), produce
`prediction/Pitch_Controllers.csv` (199 319 filas, `frame,label`). Documento técnico:
[Pitch_Controllers.md](Pitch_Controllers.md).

## Estructura

| ruta | qué es |
|---|---|
| `run.py` | único punto de entrada (`predict`, `validate`, `public`) |
| `src/config.py` | **todas** las decisiones fijas del método (semilla, modelo, decodificador) |
| `src/target_inputs.py` | lectura del ATD oficial (`.txt` + `.xml`) a metros, con el equipo de cada slot |
| `src/idsse.py` | descarga/parsing de los 7 partidos públicos IDSSE (DFL, CC BY 4.0) a 25 Hz |
| `src/labels.py` | regla de *posesión controlada* aplicada a las etiquetas DFL |
| `src/domain.py` | degradación de IDSSE para que se parezca al tracking automático del reto |
| `src/features.py` | extractor de características común (IDSSE y reto) |
| `src/decode.py` | decodificador HSMM (duración mínima + penalizaciones de cambio) |
| `src/pipeline.py` | entrenamiento, validación leave-one-match-out, inferencia |
| `src/possession_metrics.py` | copia literal del evaluador oficial |
| `data/tactical/` | datos del reto (privados, en `.gitignore`, nunca se publican) |
| `reference/` | GT público de 5 min (solo comprobación de formato/alineación) |
| `prediction/` | CSV final + `MANIFEST.json` (hash, configuración) |
| `reports/` | validación IDSSE y chequeo público (informe completo del evaluador oficial) |

`cache/` y `external/raw/` se regeneran solos: el ATD se convierte desde los archivos oficiales
y los partidos IDSSE se descargan desde una revisión fijada.

## Datos necesarios

Del paquete de datos oficial del reto solo se usan dos archivos, que deben estar en
`data/tactical/` (o en `$MTAH_DATA_DIR/tactical/`):

```text
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
```

No se usan el vídeo, la homografía, el Smart Tagging ni el broadcast.

**Los datos del reto no están en este repositorio, a propósito:** el reglamento prohíbe
redistribuirlos o publicar derivados que los reconstruyan, así que `.gitignore` excluye `data/` y
`cache/`. Quien revise el proyecto ya tiene el paquete oficial y solo tiene que copiar esos dos
archivos. El detalle de qué dato se usa, para qué y dónde vive está en
[Pitch_Controllers.md](Pitch_Controllers.md) (§2).

## Reproducir (Windows, PowerShell)

```powershell
& .\setup.ps1
& .\.venv\Scripts\python.exe .\run.py predict    # entrena con IDSSE y escribe el CSV
& .\.venv\Scripts\python.exe .\run.py validate   # evidencia de selección (IDSSE LOMO)
& .\.venv\Scripts\python.exe .\run.py public     # chequeo con el GT público: todas las métricas oficiales
```

`predict` es determinista: mismo CSV byte a byte (el hash queda en `prediction/MANIFEST.json`).
