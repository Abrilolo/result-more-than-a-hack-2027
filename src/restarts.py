"""Reglas de balón parado detectadas por algoritmo sobre el tracking (sin etiquetas).

* Saque inicial de cada parte: antes del saque el balón no está en juego, así que todo lo
  anterior es D. El saque se detecta por la formación: los dos equipos en su propio campo y casi
  quietos durante varios segundos; el saque es el instante en que esa formación se rompe. Solo
  se buscan formaciones a +-20 s de la etiqueta automática "Kick Off (Start)" del Smart Tagging;
  sin esa ancla la regla no se aplica (sin ella, en IDSSE confundió una formación con un saque).
* Balón parado en el área (córner o falta cercana): área grande llena, varios jugadores dentro
  del área chica y el grupo casi quieto durante unos segundos. Mientras dura la espera, D.
  En IDSSE esta firma es balón muerto el 99.8 % de las veces.

Las reglas se aplican después del decodificador y solo pueden convertir frames en D.
"""
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import features

FPS = 25

# saque inicial
KO_OWN_SHARE = 0.85      # fracción mínima de jugadores visibles de cada equipo en su propio campo
KO_MIN_PLAYERS = 18      # jugadores visibles mínimos
KO_STILL_MPS = 1.2       # velocidad media colectiva (1 s) por debajo de la cual el grupo está quieto
KO_MIN_SET_S = 3.0       # duración mínima de la formación de saque
KO_TAG_WINDOW_S = 20.0   # con etiqueta: buscar a +-20 s de ella

# balón parado en el área (córner o falta cercana); medidas reglamentarias del campo
BOX_DEPTH, BOX_WIDTH = 16.5, 40.32
GOAL_AREA_DEPTH, GOAL_AREA_WIDTH = 5.5, 18.32
SP_MIN_IN_BOX = 8        # jugadores (de ambos equipos) dentro del área grande
SP_MIN_IN_GOAL_AREA = 2  # y al menos estos dentro del área chica
SP_STILL_MPS = 1.0       # velocidad media colectiva (1 s) por debajo de la cual el grupo espera
SP_MIN_S = 2.0           # duración mínima de la espera
SP_BALL_FAR_M = 35.0     # si el balón se ve a más de esto de la portería, no es un balón parado ahí

def _runs(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(e[::2], e[1::2]))


def _close_gaps(mask, max_gap):
    out = mask.copy()
    runs = _runs(mask)
    for (a0, b0), (a1, _) in zip(runs[:-1], runs[1:]):
        if a1 - b0 <= max_gap:
            out[b0:a1] = True
    return out


def _collective_speed(players):
    v = np.full(players.shape[:2], np.nan, np.float32)
    v[2:-2] = np.linalg.norm(players[4:] - players[:-4], axis=2) * FPS / 4
    with np.errstate(invalid='ignore'), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        mean = np.nanmean(v, axis=1)
    return pd.Series(mean).rolling(FPS, center=True, min_periods=1).mean().to_numpy()


def kickoff_index(half, tag_index=None):
    """Índice (dentro de la parte) del saque inicial, o None si no se detecta."""
    p, team, (L, W) = half['players'], half['team'], half['pitch']
    signs = features.attack_signs(p, team, half['pitch'])
    own = []
    for k, s in zip((0, 1), signs):
        x = p[:, team == k, 0]
        vis = np.isfinite(x)
        with np.errstate(invalid='ignore'):
            own.append(((s * (x - L / 2) <= 1.0) & vis).sum(1) / np.maximum(vis.sum(1), 1))
    visible = np.isfinite(p[..., 0]).sum(1)
    formation = (own[0] >= KO_OWN_SHARE) & (own[1] >= KO_OWN_SHARE) & (visible >= KO_MIN_PLAYERS)
    still = _collective_speed(p) < KO_STILL_MPS
    setup = _close_gaps(formation & still, FPS)
    candidates = [(a, b) for a, b in _runs(setup) if b - a >= KO_MIN_SET_S * FPS]
    if tag_index is None:
        return None   # sin ancla no se aplica: en IDSSE confundió una formación a los 138 s con un saque
    w = KO_TAG_WINDOW_S * FPS
    candidates = [(a, b) for a, b in candidates if b >= tag_index - w and a <= tag_index + w]
    if not candidates:
        return None
    _, end = candidates[0]
    return int(end)


def set_piece_mask(half):
    """Espera de un balón parado en el área: córner o falta cercana.

    El ATD pierde el balón casi siempre junto al banderín, así que no se exige verlo en la
    esquina: la firma es el área grande llena, varios jugadores en el área chica y todo el grupo
    casi quieto durante unos segundos. Si el balón se ve lejos de esa portería, no cuenta.
    """
    p, b, (L, W) = half['players'], half['ball'][:, :2], half['pitch']
    still = _collective_speed(p) < SP_STILL_MPS
    out = np.zeros(len(p), bool)
    for gx in (0.0, L):
        inward = 1.0 if gx == 0.0 else -1.0
        depth = inward * (p[..., 0] - gx)                      # metros desde la línea de fondo
        lateral = np.abs(p[..., 1] - W / 2)
        in_box = ((depth >= -1) & (depth <= BOX_DEPTH) & (lateral <= BOX_WIDTH / 2)).sum(1)
        in_ga = ((depth >= -1) & (depth <= GOAL_AREA_DEPTH) & (lateral <= GOAL_AREA_WIDTH / 2)).sum(1)
        far = np.linalg.norm(b - np.array([gx, W / 2]), axis=1) > SP_BALL_FAR_M   # False si no se ve
        cand = (in_box >= SP_MIN_IN_BOX) & (in_ga >= SP_MIN_IN_GOAL_AREA) & still & ~far
        for a, e in _runs(cand):
            if e - a >= SP_MIN_S * FPS:
                out[a:e] = True
    return out


def kickoff_tags(meta_path):
    """Segundos de inicio de las etiquetas automáticas "Kick Off (Start)" del Smart Tagging."""
    root = ET.parse(meta_path).getroot()
    out = []
    for inst in root.iter('instance'):
        texts = [lab.findtext('text') or '' for lab in inst.iter('label')]
        if any(t.strip().lower() == 'kick off (start)' for t in texts):
            out.append(float(inst.findtext('start')))
    return sorted(out)


def apply(pred, half, tag_index=None, kickoff=True, set_pieces=True):
    """Convierte en D el tramo previo al saque inicial y las esperas de balón parado en el área."""
    out = pred.copy()
    info = {}
    if kickoff:
        k = kickoff_index(half, tag_index)
        info['kickoff_index'] = k
        if k is not None:
            out[:k] = 2
    if set_pieces:
        m = set_piece_mask(half)
        info['set_piece_frames'] = int(m.sum())
        info['set_pieces'] = len(_runs(m))
        out[m] = 2
    return out, info
