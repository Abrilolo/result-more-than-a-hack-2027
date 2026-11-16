"""Extractor de características por frame (25 Hz), común a las partes IDSSE y al reto.

Entrada (dict de una parte): players (T, N, 2) en metros con origen en una esquina, team (N,)
en {0 local, 1 visitante}, ball (T, 2) en metros o NaN, pitch (2,). Offline: las ventanas
centradas pueden mirar al futuro. Las características se expresan en la dirección de ataque del
local, así que intercambiar equipos (aumento de datos) es solo permutar columnas y cambiar
signos, ver `swap_teams`.
"""
import numpy as np
import pandas as pd

FPS = 25


def _nanmean(a, axis):
    with np.errstate(invalid='ignore'), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _roll(values, seconds, stat='mean'):
    s = pd.Series(values)
    w = max(1, int(round(seconds * FPS)))
    r = s.rolling(w, center=True, min_periods=1)
    return (r.mean() if stat == 'mean' else r.max() if stat == 'max' else r.min()).to_numpy(np.float32)


def attack_signs(players, team, pitch):
    """+1 si el equipo ataca hacia x creciente en esta parte (sin etiquetas)."""
    cx = [np.nanmean(players[:, team == k, 0]) for k in (0, 1)]
    home = 1.0 if cx[0] < cx[1] else -1.0
    return home, -home


def _gap_clock(seen):
    """Frames desde la última observación y hasta la siguiente (muy grande si no hay)."""
    n = len(seen)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, n)[::-1])[::-1]
    since = np.where(last >= 0, idx - last, 10 ** 6).astype(np.float32)
    until = np.where(nxt < n, nxt - idx, 10 ** 6).astype(np.float32)
    return since, until, last, nxt


def features(half):
    p = half['players'].astype(np.float32)
    team = half['team']
    b = half['ball'][:, :2].astype(np.float32)
    L, W = float(half['pitch'][0]), float(half['pitch'][1])
    T = len(b)
    s_home, s_away = attack_signs(p, team, half['pitch'])
    seen = np.isfinite(b).all(axis=1)
    f = {}
    # ---------- geometría del balón (dirección de ataque del local)
    bx = s_home * (b[:, 0] - L / 2)
    f['ball_seen'] = seen.astype(np.float32)
    f['b_x'] = bx
    f['b_absy'] = np.abs(b[:, 1] - W / 2)
    out_x = np.abs(b[:, 0] - L / 2) - L / 2
    out_y = np.abs(b[:, 1] - W / 2) - W / 2
    f['b_out'] = np.maximum(out_x, out_y)
    # velocidad suavizada del balón (solo entre frames observados a 2 de distancia)
    v = np.full(T, np.nan, np.float32)
    v[2:-2] = np.linalg.norm(b[4:] - b[:-4], axis=1) * FPS / 4
    f['b_speed'] = v
    bvel = np.full_like(b, np.nan)
    bvel[2:-2] = (b[4:] - b[:-4]) * FPS / 4
    ctrl = {}
    # ---------- geometría de los jugadores
    vel = np.full_like(p, np.nan)
    vel[2:-2] = (p[4:] - p[:-4]) * FPS / 4
    speed = np.linalg.norm(vel, axis=2)
    for k, name, s in ((0, 'h', s_home), (1, 'a', s_away)):
        q = p[:, team == k]
        sp = speed[:, team == k]
        vx = vel[:, team == k, 0] * s
        present = np.isfinite(q).all(axis=2)
        f[f'n_{name}'] = present.sum(axis=1).astype(np.float32)
        d = np.linalg.norm(q - b[:, None], axis=2)
        d = np.where(np.isfinite(d), d, np.inf)
        ds = np.sort(d, axis=1)
        for r in range(2):
            col = ds[:, r] if ds.shape[1] > r else np.full(T, np.inf)
            f[f'd{r + 1}_{name}'] = np.where(np.isfinite(col) & seen, col, np.nan).astype(np.float32)
        for radius in (2, 5, 10):
            f[f'c{radius}_{name}'] = np.where(seen, (d <= radius).sum(axis=1), np.nan).astype(np.float32)
        nearest = np.argmin(d, axis=1)
        sp_near = np.take_along_axis(np.where(np.isfinite(sp), sp, np.nan), nearest[:, None], 1)[:, 0]
        f[f'near_speed_{name}'] = np.where(seen, sp_near, np.nan).astype(np.float32)
        # control: jugador más cercano pegado al balón y el balón moviéndose con él
        v_near = np.take_along_axis(vel[:, team == k], nearest[:, None, None], 1)[:, 0]
        rel = np.linalg.norm(bvel - v_near, axis=1)
        f[f'rel_{name}'] = np.where(seen, rel, np.nan).astype(np.float32)
        ctrl[k] = seen & (ds[:, 0] < 1.5) & (rel < 3.0)
        # forma colectiva en la dirección de ataque de cada equipo
        xa = s * (q[:, :, 0] - L / 2)
        f[f'adv_{name}'] = _nanmean(xa, 1)
        f[f'sx_{name}'] = np.nanstd(xa, axis=1) if q.shape[1] else np.nan
        f[f'sy_{name}'] = np.nanstd(q[:, :, 1], axis=1) if q.shape[1] else np.nan
        f[f'front_{name}'] = np.nanmax(np.where(np.isfinite(xa), xa, -np.inf), axis=1)
        f[f'back_{name}'] = np.nanmin(np.where(np.isfinite(xa), xa, np.inf), axis=1)
        f[f'spd_{name}'] = _nanmean(sp, 1)
        f[f'vx_{name}'] = _nanmean(vx, 1)
        f[f'fast_{name}'] = (np.where(np.isfinite(sp), sp, 0) > 4).sum(axis=1).astype(np.float32)
    for key in ('front_h', 'front_a', 'back_h', 'back_a'):
        f[key] = np.where(np.isfinite(f[key]), f[key], np.nan).astype(np.float32)
    f['margin'] = f['d1_h'] - f['d1_a']
    f['spd_all'] = _nanmean(np.stack([f['spd_h'], f['spd_a']]), 0)
    # balón respecto al bloque de cada equipo: cuánto ha penetrado en el bloque rival
    f['b_minus_back_a'] = f['b_x'] + f['back_a']   # el sistema del visitante está reflejado: back_a en su sistema
    f['b_minus_back_h'] = -f['b_x'] + f['back_h']
    # último / próximo equipo en control (cruza huecos del balón y balones sueltos)
    c = np.where(ctrl[0] & ~ctrl[1], 1.0, np.where(ctrl[1] & ~ctrl[0], -1.0, 0.0))
    f['ctrl'] = c.astype(np.float32)
    has = c != 0
    since, until, last, nxt = _gap_clock(has)
    f['ctrl_last'] = np.where(last >= 0, c[np.clip(last, 0, T - 1)], 0).astype(np.float32)
    f['ctrl_next'] = np.where(nxt < T, c[np.clip(nxt, 0, T - 1)], 0).astype(np.float32)
    f['ctrl_since'] = (np.minimum(since, 1500) / FPS).astype(np.float32)
    f['ctrl_until'] = (np.minimum(until, 1500) / FPS).astype(np.float32)
    table = pd.DataFrame(f)
    for sec in (1, 3, 8):
        table[f'ctrl_m{sec}'] = _roll(c, sec)
        table[f'ctrlabs_m{sec}'] = _roll(np.abs(c), sec)
    # ---------- contexto temporal (centrado, offline)
    for name in ('margin', 'b_speed', 'spd_all', 'vx_h', 'vx_a', 'adv_h', 'adv_a', 'c5_h', 'c5_a',
                 'near_speed_h', 'near_speed_a', 'b_x', 'b_out', 'fast_h', 'fast_a'):
        for sec in (1, 4):
            table[f'{name}_m{sec}'] = _roll(table[name].to_numpy(), sec)
    for sec in (12,):
        for name in ('spd_all', 'vx_h', 'vx_a', 'margin', 'adv_h', 'adv_a'):
            table[f'{name}_m{sec}'] = _roll(table[name].to_numpy(), sec)
    # cambio de velocidad colectiva: paradas / reanudaciones
    table['spd_all_d2'] = _roll(table['spd_all'].to_numpy(), 2).astype(np.float32)
    table['spd_all_fwd'] = np.r_[table['spd_all_d2'].to_numpy()[50:], np.full(50, np.nan)] - table['spd_all_d2'].to_numpy()
    for sec in (1, 4, 12):
        table[f'seen_m{sec}'] = _roll(seen.astype(np.float32), sec)
    # ---------- puente sobre huecos del balón: estado en los extremos del hueco actual
    since, until, last, nxt = _gap_clock(seen)
    table['gap_since'] = np.minimum(since, 1500) / FPS
    table['gap_until'] = np.minimum(until, 1500) / FPS
    m = table['margin_m1'].to_numpy()
    idx_last = np.clip(last, 0, T - 1)
    idx_next = np.clip(nxt, 0, T - 1)
    table['margin_last'] = np.where(last >= 0, m[idx_last], np.nan)
    table['margin_next'] = np.where(nxt < T, m[idx_next], np.nan)
    table['bx_last'] = np.where(last >= 0, bx[idx_last], np.nan)
    table['bx_next'] = np.where(nxt < T, bx[idx_next], np.nan)
    table['bout_last'] = np.where(last >= 0, f['b_out'][idx_last], np.nan)
    table['bout_next'] = np.where(nxt < T, f['b_out'][idx_next], np.nan)
    return table.replace([np.inf, -np.inf], np.nan).astype(np.float32)


# columnas que se intercambian entre equipos y columnas que cambian de signo al reflejar
_PAIR = {}
_FLIP = {'b_x', 'b_x_m1', 'b_x_m4', 'bx_last', 'bx_next'}
_NEG = {'margin', 'margin_m1', 'margin_m4', 'margin_m12', 'margin_last', 'margin_next',
        'ctrl', 'ctrl_last', 'ctrl_next', 'ctrl_m1', 'ctrl_m3', 'ctrl_m8'}


def swap_teams(table):
    """Los mismos frames vistos con local y visitante intercambiados (aumento de datos)."""
    out = table.copy()
    for col in table.columns:
        if col.endswith('_h') or '_h_' in col:
            other = col.replace('_h', '_a') if col.endswith('_h') else col.replace('_h_', '_a_')
            if other in table.columns:
                out[col], out[other] = table[other].to_numpy(), table[col].to_numpy()
    for col in _FLIP | _NEG:
        if col in out.columns:
            out[col] = -out[col]
    # b_minus_back_* ya quedan bien con el intercambio; los signos son consistentes
    return out
