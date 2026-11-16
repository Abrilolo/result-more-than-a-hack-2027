"""Todas las decisiones fijas del método, en un solo lugar.

Cada valor se eligió con el spellF1 leave-one-match-out sobre los 7 partidos IDSSE (ver
reports/VALIDATION_IDSSE.json y Pitch_Controllers.md), nunca con el partido del reto.
"""
TEAM = 'Pitch_Controllers'  # equipo Pitch Controllers: CSV en prediction/<TEAM>.csv
SEED = 2026
CONTROL_S = 2.0          # regla de posesión controlada aplicada a las etiquetas DFL
FEATURE_VARIANT = 'v2'
AUG_DRAWS = 3            # degradaciones independientes de cada parte IDSSE usadas para entrenar
TRAIN_STRIDE = 5         # una fila de entrenamiento cada 5 frames de cada degradación
XGB = dict(n_estimators=900, max_depth=8, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7,
           min_child_weight=10, reg_lambda=2.0)
DECODER = dict(min_poss_s=2.0, min_dead_s=1.0, pen_team=12.0, pen_dead=40.0, temp=1.0)
