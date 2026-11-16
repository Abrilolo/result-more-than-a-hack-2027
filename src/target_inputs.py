"""Entradas del reto: el ATD automático oficial (jugadores con equipo, balón).

Lee tres archivos del paquete de datos oficial, sin ninguna etiqueta:

    data/tactical/BAR-ATH_tactical_atd.xml   metadatos: equipo de cada slot de tracking
    data/tactical/BAR-ATH_tactical_atd.txt   posiciones normalizadas 0-1 por frame

y el Smart Tagging automático (`BAR-ATH_tactical_smart_tagging.xml`), del que solo se toma la
hora de la etiqueta "Kick Off (Start)" como ancla para buscar el saque inicial. Si falta alguno
de los tres, se detiene con un mensaje que dice dónde copiarlo (nunca sigue sin él).

`MTAH_DATA_DIR` (la misma variable que usa el repositorio del reto) permite apuntar a otra
carpeta que contenga `tactical/`. El resultado se guarda en `cache/atd.npz`; coordenadas en
metros con origen en una esquina del campo: x en [0, 105.3], y en [0, 68.0].
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PITCH = np.array([105.3, 68.0])
N_FRAMES = 199319
HALVES = ((1, 11185, 90651), (2, 110808, 190197))
ROOT = Path(__file__).resolve().parents[1]
GAME = 'BAR-ATH'


def data_dir():
    return Path(os.environ.get('MTAH_DATA_DIR', ROOT / 'data'))


INPUT_FILES = (f'{GAME}_tactical_atd.txt', f'{GAME}_tactical_atd.xml', f'{GAME}_tactical_smart_tagging.xml')


def _required(name):
    path = data_dir() / 'tactical' / name
    if not path.exists():
        raise FileNotFoundError(
            f'Falta {path}.\nCopia en {data_dir() / "tactical"} estos archivos del paquete de datos oficial '
            f'(o define MTAH_DATA_DIR con la carpeta que contiene tactical/): ' + ', '.join(INPUT_FILES))
    return path


def atd_paths():
    return _required(f'{GAME}_tactical_atd.xml'), _required(f'{GAME}_tactical_atd.txt')


def team_of_each_slot(meta_path):
    """0 = local, 1 = visitante para cada slot. El primer equipo del XML es el local
    (convención FIFA/EPTS; en este partido, Barcelona)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get('id') for team in root.iter('Team')]
    assert len(team_ids) == 2, 'se esperaban 2 equipos en los metadatos del ATD'
    return np.array([0 if p.get('teamId') == team_ids[0] else 1 for p in root.iter('Player')], np.int8)


def build_atd():
    """Lee el ATD oficial. Formato de cada línea: `frame:x,y,v;x,y,v;...:x,y,v` (balón al final),
    `NaN` donde no hay observación; `frame` empieza en 1 (frame de vídeo = frame - 1)."""
    meta, raw = atd_paths()
    team = team_of_each_slot(meta)
    players = np.full((N_FRAMES, len(team), 2), np.nan, np.float32)
    ball = np.full((N_FRAMES, 2), np.nan, np.float32)
    with open(raw) as handle:
        for line in handle:
            parts = line.split(':')
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            for slot, group in enumerate(parts[1].split(';')):
                if group[0] != 'N':
                    x, y, _ = group.split(',')
                    players[frame, slot] = float(x), float(y)
            if parts[2][0] != 'N':
                x, y = parts[2].split(',')[:2]
                ball[frame] = float(x), float(y)
    scale = PITCH.astype(np.float32)
    return {'players': players * scale, 'team': team, 'ball': ball * scale}


def load_atd(cache_dir):
    """ATD en metros, generado desde los archivos oficiales y guardado en caché."""
    path = Path(cache_dir) / 'atd.npz'
    if not path.exists():
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **build_atd())
    with np.load(path, allow_pickle=False) as z:
        return {'players': z['players'].astype(np.float32), 'team': z['team'].astype(np.int8),
                'ball': z['ball'].astype(np.float32)}


def smart_tagging_path():
    return _required(f'{GAME}_tactical_smart_tagging.xml')
