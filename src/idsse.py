"""Partidos externos de entrenamiento IDSSE (DFL / Bassek et al. 2025, CC BY 4.0).

Lee los XML públicos de Sportec a 25 Hz completos y los pasa a la misma representación que el
reto: metros con origen en una esquina del campo, jugadores con id de equipo (0 = local,
1 = visitante), balón (x, y) y una etiqueta por frame (0 = H, 1 = A, 2 = D, -1 = desconocida).
"""
from pathlib import Path

import numpy as np
from lxml import etree

REVISION = 'a715a38dfbaf5f58e431727c2b78d174101a703c'
MATCHES = ['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ', 'J03WR9']
PERIODS = {'firstHalf': 1, 'secondHalf': 2}


def raw_paths(raw_dir, match):
    comp = '000001' if match in ('J03WMX', 'J03WN1') else '000002'
    raw_dir = Path(raw_dir)
    return (raw_dir / f'DFL_02_01_matchinformation_DFL-COM-{comp}_DFL-MAT-{match}.xml',
            raw_dir / f'DFL_04_03_positions_raw_observed_DFL-COM-{comp}_DFL-MAT-{match}.xml')


def download(raw_dir, match):
    from huggingface_hub import hf_hub_download
    for path in raw_paths(raw_dir, match):
        if not path.exists():
            hf_hub_download(repo_id='pysport/idsse-data', repo_type='dataset', revision=REVISION,
                            filename=path.name, local_dir=str(raw_dir))


def parse(raw_dir, match):
    meta_path, tracking_path = raw_paths(raw_dir, match)
    root = etree.parse(str(meta_path))
    env = root.find('.//Environment')
    pitch = np.array([float(env.get('PitchX')), float(env.get('PitchY'))])
    teams = {t.get('TeamId'): {'home': 0, 'guest': 1}[t.get('Role')]
             for t in root.findall('.//Teams/Team') if t.get('Role') in ('home', 'guest')}
    groups, current = {}, None
    for event, elem in etree.iterparse(str(tracking_path), events=('start', 'end'), huge_tree=True):
        tag = etree.QName(elem).localname
        if event == 'start':
            if tag == 'FrameSet':
                current = (PERIODS.get(elem.get('GameSection')), elem.get('TeamId'), elem.get('PersonId'), [])
            continue
        if tag == 'Frame' and current is not None:
            half, team, _, rows = current
            if half and (team == 'BALL' or team in teams):
                if team == 'BALL':
                    status, owner = elem.get('BallStatus'), elem.get('BallPossession')
                    y = 2 if status == '0' else (int(owner) - 1 if status == '1' and owner in ('1', '2') else -1)
                    rows.append((int(elem.get('N')), float(elem.get('X')), float(elem.get('Y')),
                                 float(elem.get('Z', 'nan')), y))
                else:
                    rows.append((int(elem.get('N')), float(elem.get('X')), float(elem.get('Y'))))
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        elif tag == 'FrameSet' and current is not None:
            half, team, person, rows = current
            if half and rows and (team == 'BALL' or team in teams):
                group = groups.setdefault(half, {'ball': None, 'people': []})
                if team == 'BALL':
                    group['ball'] = np.asarray(rows, dtype=float)
                else:
                    group['people'].append((teams[team], np.asarray(rows, dtype=float)))
            current = None
            elem.clear()
    out = {'pitch': pitch}
    for half in (1, 2):
        g = groups[half]
        b = g['ball']
        frames = np.arange(int(b[:, 0].min()), int(b[:, 0].max()) + 1)
        n = len(frames)
        ball = np.full((n, 3), np.nan, np.float32)
        y = np.full(n, -1, np.int8)
        idx = (b[:, 0] - frames[0]).astype(int)
        ball[idx] = b[:, 1:4]
        y[idx] = b[:, 4].astype(int)
        players = np.full((n, len(g['people']), 2), np.nan, np.float32)
        team_ids = np.zeros(len(g['people']), np.int8)
        for j, (team, a) in enumerate(g['people']):
            take = (a[:, 0] >= frames[0]) & (a[:, 0] <= frames[-1])
            a = a[take]
            players[(a[:, 0] - frames[0]).astype(int), j] = a[:, 1:3]
            team_ids[j] = team
        # origen en el centro -> origen en la esquina, misma convención que el ATD del reto
        ball[:, :2] += pitch / 2
        players += pitch / 2
        out[f'ball_{half}'] = ball
        out[f'players_{half}'] = players
        out[f'team_{half}'] = team_ids
        out[f'y_{half}'] = y
    return out


def load(cache_dir, raw_dir, match):
    """Lectura con caché; la caché es función pura de los XML públicos fijados."""
    path = Path(cache_dir) / f'idsse_{match}.npz'
    if not path.exists():
        download(raw_dir, match)
        np.savez_compressed(path, **parse(raw_dir, match))
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    halves = []
    for half in (1, 2):
        halves.append({'match': match, 'half': half, 'ball': data[f'ball_{half}'],
                       'players': data[f'players_{half}'], 'team': data[f'team_{half}'],
                       'y': data[f'y_{half}'], 'pitch': data['pitch']})
    return halves
