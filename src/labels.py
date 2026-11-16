"""Convierte la posesión cruda de DFL a la convención de posesión *controlada* del reto.

`BallPossession` de DFL cambia con cada toque. El reto solo cambia la posesión cuando el rival
gana el *control* ("un simple toque sin control", "un desvío" y "un duelo aéreo ganado sin
control posterior" no cuentan; "un jugador conserva el balón unos segundos").
Regla, en una sola pasada hacia delante: una racha del equipo sin posesión más corta que
`control_s` segundos se asigna al equipo en posesión. El balón muerto (2) nunca se modifica, y el
primer equipo tras una reanudación es el poseedor sea cual sea la duración de su racha (una
reanudación tiene un poseedor claro).
"""
import numpy as np

FPS = 25


def runs(y):
    e = np.flatnonzero(np.diff(np.r_[-99, y, -99]))
    return list(zip(e[:-1], e[1:], y[e[:-1]]))


def controlled(y, control_s=2.0):
    out = y.copy()
    min_len = int(round(control_s * FPS))
    current = None
    for a, b, k in runs(y):
        if k not in (0, 1):
            current = None
            continue
        if current is None or k == current or b - a >= min_len:
            current = k
        else:
            out[a:b] = current
    return out
