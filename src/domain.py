"""Hace que las partes limpias de IDSSE se parezcan al tracking automático del reto (sin etiquetas).

* Disponibilidad del balón: el patrón exacto de presencia/ausencia del balón del reto (una
  ventana aleatoria de una parte del reto), para que la longitud y frecuencia de los huecos
  coincidan con lo que verá la inferencia.
* Error de posición del balón y de los jugadores, y jugadores que desaparecen en bloques
  aleatorios, para que cada equipo muestre en promedio tantos jugadores como el ATD del reto.
"""
import numpy as np


def ball_mask_window(masks, length, rng):
    """Una ventana contigua de una máscara de disponibilidad del reto (repetida si es corta)."""
    m = masks[rng.integers(len(masks))]
    if len(m) < length:
        m = np.tile(m, length // len(m) + 1)
    start = rng.integers(0, len(m) - length + 1)
    return m[start:start + length]


def player_dropout(T, N, rng, missing=0.07, mean_block_s=3.0):
    """Desaparición de Markov de dos estados por jugador, con proporción estacionaria `missing`."""
    p_off = 1.0 / (mean_block_s * 25)              # salir de un bloque ausente
    p_on = p_off * missing / (1 - missing)         # entrar en un bloque ausente
    out = np.zeros((T, N), bool)
    state = rng.random(N) < missing
    u = rng.random((T, N))
    for t in range(T):
        flip = np.where(state, u[t] < p_off, u[t] < p_on)
        state = state ^ flip
        out[t] = state
    return out


def smooth_noise(shape, sigma, rng, seconds=2.0):
    """Error de posición de baja frecuencia (deriva del tracking), no ruido blanco: el ruido
    blanco inflaría las velocidades entre frames muy por encima de las del ATD del reto."""
    T = shape[0]
    k = max(1, int(seconds * 25))
    knots = rng.normal(0, sigma, (T // k + 2,) + shape[1:])
    t = np.arange(T) / k
    i = t.astype(int)
    w = (t - i).reshape((T,) + (1,) * (len(shape) - 1))
    return ((1 - w) * knots[i] + w * knots[i + 1]).astype(np.float32)


def degrade(half, ball_masks, seed, ball_noise=0.7, player_noise=0.4, missing=0.06):
    rng = np.random.default_rng(seed)
    out = dict(half)
    T = len(half['ball'])
    ball = half['ball'][:, :2].copy()
    ball += smooth_noise(ball.shape, ball_noise, rng, 1.0)
    ball[~ball_mask_window(ball_masks, T, rng)] = np.nan
    players = half['players'].copy()
    players += smooth_noise(players.shape, player_noise, rng, 2.0)
    players[player_dropout(T, players.shape[1], rng, missing)] = np.nan
    out['ball'], out['players'] = ball, players
    return out
