"""Decodificador de segmentos: Viterbi con duración mínima explícita (HSMM como cadenas de
estados) sobre H, A, D.

Las emisiones son las probabilidades del clasificador divididas por la proporción de clases del
entrenamiento (log-verosimilitudes escaladas). Cada etiqueta k es una cadena de `min_len[k]`
subestados que hay que recorrer antes de salir, así que ningún segmento decodificado es más corto
que su mínimo. Salir cuesta una penalización que depende del par (equipo<->equipo o
vivo<->muerto). Se decodifica en bloques de `STEP` frames; después las fronteras se ajustan al
frame exacto con la evidencia a 25 Hz.
"""
import numpy as np

STEP = 5
NEG = -1e18


def block_logp(logp, step=STEP):
    T = len(logp)
    pad = (-T) % step
    if pad:
        logp = np.vstack([logp, np.repeat(logp[-1:], pad, axis=0)])
    return logp.reshape(-1, step, logp.shape[1]).sum(axis=1)


def viterbi_min_duration(em, min_len, penalty):
    """em: (T, K) log-puntuaciones; min_len: (K,) pasos >= 1; penalty[j, k]: coste de j -> k."""
    T, K = em.shape
    last = np.asarray(min_len, int) - 1
    D = int(last.max()) + 1
    rows = np.arange(K)
    valid = np.arange(D)[None, :] <= last[:, None]
    pen = np.asarray(penalty, float).copy()
    np.fill_diagonal(pen, np.inf)
    score = np.full((K, D), NEG)
    score[:, 0] = em[0]
    enter_from = np.zeros((T, K), np.int8)
    stay = np.zeros((T, K), bool)
    for t in range(1, T):
        exit_score = score[rows, last]
        cand = exit_score[:, None] - pen
        j_best = np.argmax(cand, axis=0)
        entry = cand[j_best, rows]
        new = np.full((K, D), NEG)
        new[:, 1:] = score[:, :-1]
        advanced = new[rows, last]            # NEG cuando last == 0
        s = exit_score >= advanced
        new[rows, last] = np.where(s, exit_score, advanced)
        single = last == 0
        # min_len 1: el subestado 0 es también el último; la entrada compite con el auto-bucle
        take_entry = single & (entry > new[rows, 0])
        stay[t] = s & ~take_entry
        new[:, 0] = np.where(single, np.where(take_entry, entry, new[:, 0]), entry)
        enter_from[t] = j_best
        new += em[t][:, None]
        new[~valid] = NEG
        score = new
    path = np.empty(T, np.int8)
    k = int(np.argmax(score[rows, last]))
    i = int(last[k])
    for t in range(T - 1, -1, -1):
        path[t] = k
        if t == 0:
            break
        if i == last[k] and stay[t, k]:
            continue
        if i > 0:
            i -= 1
        else:
            k = int(enter_from[t, k])
            i = int(last[k])
    return path


def refine(path_blocks, logp, step=STEP):
    """Expande el camino por bloques y mueve cada frontera al mejor frame dentro de +-step."""
    T = len(logp)
    labels = np.repeat(path_blocks, step)[:T].astype(np.int8)
    cum = np.vstack([np.zeros((1, logp.shape[1])), np.cumsum(logp, axis=0)])
    for e in np.flatnonzero(np.diff(labels)) + 1:
        a, b = labels[e - 1], labels[e]
        lo, hi = max(1, e - step), min(T - 1, e + step)
        if labels[lo - 1] != a or labels[hi] != b:
            continue  # otra frontera demasiado cerca; se mantiene la del bloque
        c = np.arange(lo, hi + 1)
        gain = (cum[c, a] - cum[lo, a]) + (cum[hi + 1, b] - cum[c, b])
        best = int(c[np.argmax(gain)])
        labels[lo:best] = a
        labels[best:hi + 1] = b
    return labels


def decode_half(prob, prior, params):
    """prob: (T, 3) probabilidades a 25 Hz -> etiquetas 0/1/2 por frame."""
    logp = params.get('temp', 1.0) * (np.log(np.clip(prob, 1e-6, 1)) - np.log(prior))
    em = block_logp(logp)
    steps_per_s = 25 / STEP
    min_len = np.maximum(1, np.round(np.array([params['min_poss_s'], params['min_poss_s'],
                                               params['min_dead_s']]) * steps_per_s)).astype(int)
    pt, pd = params['pen_team'], params['pen_dead']
    penalty = np.array([[0, pt, pd], [pt, 0, pd], [pd, pd, 0]], float)
    return refine(viterbi_min_duration(em, min_len, penalty), logp)
