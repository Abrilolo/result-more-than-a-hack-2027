"""Posesión H/A/D de BAR-ATH a partir del tracking automático. Un comando, determinista.

    python run.py predict     # entrena con IDSSE (los 7 partidos) y escribe el CSV de entrega
    python run.py validate    # spellF1 leave-one-match-out en IDSSE (evidencia de selección)
    python run.py public      # chequeo de sanidad del CSV con el GT público de 5 min (sin ajustar nada)
"""
import argparse
import hashlib
import json
import sys
sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import config  # noqa: E402
import pipeline as pl  # noqa: E402
import possession_metrics as pm  # noqa: E402
import target_inputs as ti  # noqa: E402
import decode  # noqa: E402

OUT = ROOT / 'prediction'
CSV = OUT / f'{config.TEAM}.csv'
PUBLIC_GT = ROOT / 'reference' / 'BAR-ATH_possession_gt_public.csv'
LABELS = np.array(list('HADX'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict():
    pl.CACHE.mkdir(exist_ok=True)
    masks = pl.ball_masks()
    parts = pl.idsse_tables(masks)
    model, prior, columns = pl.fit(parts)
    labels = np.full(ti.N_FRAMES, 'X', dtype='<U1')
    stats = []
    for half in pl.target_halves():
        x = pl.target_features(half)
        assert list(x.columns) == columns
        prob = pl.predict(model, x)
        pred = decode.decode_half(prob, prior, config.DECODER)
        labels[half['start']:half['end'] + 1] = LABELS[pred]
        stats.append({'half': half['half'], 'frames': len(pred),
                      'share': {k: round(float((pred == i).mean()), 4) for i, k in enumerate('HAD')}})
    OUT.mkdir(exist_ok=True)
    pd.DataFrame({'frame': np.arange(ti.N_FRAMES), 'label': labels}).to_csv(CSV, index=False, lineterminator='\n')
    assert pm.validate_submission(pm.read_label_csv(CSV), ti.N_FRAMES) is None
    manifest = {'csv': CSV.name, 'sha256': sha(CSV), 'halves': stats, 'decoder': config.DECODER,
                'control_s': config.CONTROL_S, 'aug_draws': config.AUG_DRAWS, 'train_stride': config.TRAIN_STRIDE,
                'feature_variant': config.FEATURE_VARIANT, 'xgb': config.XGB, 'seed': config.SEED,
                'training_matches': pl.idsse.MATCHES}
    (OUT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps(manifest, indent=2))


def validate():
    parts = pl.idsse_tables(pl.ball_masks())
    probs, prior = pl.oof(parts)
    result = pl.score(parts, probs, prior, config.DECODER)
    report = {'protocol': 'leave-one-match-out sobre 7 partidos IDSSE, degradados al ruido del reto, '
                          'etiquetas = posesión DFL con la regla de posesión controlada',
              'decoder': config.DECODER, **result}
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports' / 'VALIDATION_IDSSE.json').write_text(json.dumps(report, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps(report, indent=2))


def public():
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    pred = pm.label_codes(pm.read_label_csv(CSV))
    r = pm.evaluate(gt, pred, bootstrap_n=0)
    keys = ['spell_f1_mean', 'macro_f1', 'accuracy', 'n_gt_spells', 'n_pred_spells', 'n_abstain_on_graded']
    out = {k: r[k] for k in keys}
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports' / 'PUBLIC_SANITY_CHECK.json').write_text(json.dumps(out, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['predict', 'validate', 'public'])
    {'predict': predict, 'validate': validate, 'public': public}[parser.parse_args().command]()
