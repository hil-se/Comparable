import numpy as np
import pandas as pd


def generate_pairs(df):
    """
    Build all ordered pairwise comparisons using vectorized NumPy operations.
    Expected columns: A (protected attribute), Y (target), Y_pred (prediction score).
    """
    a_vals = df["A"].to_numpy()
    y_vals = df["Y"].to_numpy()
    pred_vals = df["Y_pred"].to_numpy()

    rating_delta = y_vals[:, None] - y_vals[None, :]
    pred_delta = pred_vals[:, None] - pred_vals[None, :]

    labels = np.sign(rating_delta).astype(np.int8)
    preds = np.sign(pred_delta).astype(np.int8)

    n = len(df)
    idx_a, idx_b = np.indices((n, n))

    return pd.DataFrame(
        {
            "A": a_vals[idx_a.ravel()],
            "B": a_vals[idx_b.ravel()],
            "Label": labels.ravel(),
            "pred": preds.ravel(),
        }
    )
