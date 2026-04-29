"""
predictor.py – Single-sample prediction helper.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def predict_single(model, input_data: Dict[str, Any]) -> Tuple[float, int]:
    """
    Run inference on a single patient record.

    Parameters
    ----------
    model      : fitted sklearn-compatible model
    input_data : dict with keys matching feature names

    Returns
    -------
    prob  : float – probability of disease
    label : int   – 0 or 1
    """
    from src.data_loader import NUM_COLS, CAT_COLS

    ordered_cols = NUM_COLS + CAT_COLS
    row = [input_data.get(c, 0) for c in ordered_cols]
    X = np.array(row, dtype=float).reshape(1, -1)

    prob = float(model.predict_proba(X)[0][1])
    label = int(prob >= 0.5)
    return prob, label
