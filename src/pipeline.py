"""Bloques comunes: datos, entrenamiento, predicción fuera de pliegue y puntuación de posesiones."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import decode  # noqa: E402
import domain  # noqa: E402
import features  # noqa: E402
import idsse  # noqa: E402
import labels  # noqa: E402
import possession_metrics as pm  # noqa: E402

CACHE = ROOT / 'cache'
RAW = ROOT / 'external' / 'raw'
IDSSE = ROOT / 'external' / 'idsse'   # partidos IDSSE ya procesados (se publican en el repo)
MODEL = ROOT / 'model'
from config import AUG_DRAWS, CONTROL_S, TRAIN_STRIDE, XGB, SEED, FEATURE_VARIANT  # noqa: E402


def idsse_tables(ball_masks, variant=FEATURE_VARIANT, draws=None):
    """Tablas de características degradadas + etiquetas controladas de cada parte IDSSE (con caché).

    Cada degradación es independiente (otra ventana de huecos del balón, otro ruido, otras
    desapariciones) sobre la misma parte. La 0 es la que se puntúa en la validación; todas se
    usan para entrenar.
    """
    draws = AUG_DRAWS if draws is None else draws
    out = []
    for mi, match in enumerate(idsse.MATCHES):
        for half in idsse.load(IDSSE, RAW, match):
            for draw in range(draws):
                suffix = f'_d{draw}' if draw else ''
                path = CACHE / f'feat_{variant}_{match}_{half["half"]}{suffix}.npz'
                if path.exists():
                    with np.load(path, allow_pickle=False) as z:
                        x = pd.DataFrame(z['x'], columns=z['columns'].tolist())
                        y = z['y']
                else:
                    seed = SEED + 100 * mi + half['half'] + 10000 * draw
                    x = features.features(domain.degrade(half, ball_masks, seed))
                    y = labels.controlled(half['y'], CONTROL_S)
                    np.savez_compressed(path, x=x.to_numpy(np.float32),
                                        columns=np.array(list(x.columns), dtype='U'), y=y)
                out.append({'match': match, 'half': half['half'], 'draw': draw, 'x': x, 'y': y})
    return out


def training_matrix(parts):
    xs, ys = [], []
    for p in parts:
        keep = np.flatnonzero(p['y'] >= 0)[::TRAIN_STRIDE]
        x = p['x'].iloc[keep]
        y = p['y'][keep].astype(int)
        xs += [x, features.swap_teams(x)]
        ys += [y, np.where(y == 2, 2, 1 - y)]
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def fit(parts, n_jobs=20):
    x, y = training_matrix(parts)
    prior = np.bincount(y, minlength=3) / len(y)
    model = XGBClassifier(**XGB, objective='multi:softprob', num_class=3, tree_method='hist',
                          n_jobs=n_jobs, random_state=SEED, eval_metric='mlogloss')
    model.fit(x, y)
    return model, prior, list(x.columns)


def predict(model, x):
    """Probabilidad simetrizada: promedio de la vista directa y la vista con equipos intercambiados."""
    p = model.predict_proba(x)
    q = model.predict_proba(features.swap_teams(x))[:, [1, 0, 2]]
    return (p + q) / 2


def oof(parts):
    probs = {}
    for match in idsse.MATCHES:
        model, prior, _ = fit([p for p in parts if p['match'] != match])
        for p in parts:
            if p['match'] == match and p['draw'] == 0:
                probs[(match, p['half'])] = predict(model, p['x'])
        print('oof', match, flush=True)
    return probs, prior


def score(parts, probs, prior, params):
    """spellF1 oficial por partido IDSSE (las dos partes juntas, respetando el descanso), promediado entre partidos."""
    by_match = {}
    for p in parts:
        if p['draw']:
            continue
        pred = decode.decode_half(probs[(p['match'], p['half'])], prior, params)
        by_match.setdefault(p['match'], []).append((p['y'].astype(np.int64), pred.astype(np.int64)))
    per_match, ratios, accs = {}, [], []
    for match, halves in by_match.items():
        y = np.concatenate([h[0] for h in halves])
        pred = np.concatenate([h[1] for h in halves])
        bounds = np.cumsum([0] + [len(h[0]) for h in halves])
        r = pm.spell_scores(y, pred, list(zip(bounds[:-1], bounds[1:])))
        per_match[match] = round(r['mean_f1'], 4)
        ratios.append(r['n_pred'] / r['n_gt'])
        accs.append(float((y == pred).mean()))
    return {'spell_f1': float(np.mean(list(per_match.values()))), 'acc': float(np.mean(accs)),
            'pred_per_gt': float(np.mean(ratios)), 'per_match': per_match}


def target_halves():
    """Partes evaluadas del reto a partir del ATD automático oficial (jugadores con equipo, balón)."""
    import target_inputs as ti
    atd = ti.load_atd(CACHE)
    out = []
    for half, a, b in ti.HALVES:
        out.append({'half': half, 'start': a, 'end': b, 'players': atd['players'][a:b + 1],
                    'team': atd['team'], 'ball': atd['ball'][a:b + 1], 'pitch': ti.PITCH})
    return out


def ball_masks():
    return [np.isfinite(h['ball'][:, 0]) for h in target_halves()]


def target_features(half):
    return features.features(half)


def save_model(model, prior, columns):
    """Guarda el modelo entrenado y lo necesario para usarlo (proporción de clases, columnas)."""
    MODEL.mkdir(exist_ok=True)
    model.save_model(MODEL / 'model.ubj')
    meta = {'prior': [float(v) for v in prior], 'columns': columns,
            'model_sha256': hashlib.sha256((MODEL / 'model.ubj').read_bytes()).hexdigest()}
    (MODEL / 'MODEL.json').write_text(json.dumps(meta, indent=2), encoding='utf-8', newline='\n')
    return meta


def load_model():
    meta_path = MODEL / 'MODEL.json'
    if not meta_path.exists():
        raise FileNotFoundError('No hay modelo en model/: ejecuta primero `python run.py train`')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    digest = hashlib.sha256((MODEL / 'model.ubj').read_bytes()).hexdigest()
    assert digest == meta['model_sha256'], 'model/model.ubj no coincide con model/MODEL.json'
    model = XGBClassifier()
    model.load_model(MODEL / 'model.ubj')
    return model, np.array(meta['prior']), meta['columns']
