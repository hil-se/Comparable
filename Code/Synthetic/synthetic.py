import contextlib
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Suppress TensorFlow C++ INFO/WARNING logs (e.g., local_rendezvous messages).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

import Classification
import DataProcessing
from metrics import Metrics

PAIRWISE_DECISION_THRESHOLD = 0.0
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "Data"
RESULTS_DIR = BASE_DIR / "Results"
HISTOGRAM_DIR = RESULTS_DIR / "Histograms"
TRAINING_LOG_DIR = RESULTS_DIR / "TrainingLogs"


class _TeeStream:
    def __init__(self, console_stream, log_stream):
        self.console_stream = console_stream
        self.log_stream = log_stream

    def write(self, data):
        self.console_stream.write(data)
        self.log_stream.write(data)
        return len(data)

    def flush(self):
        self.console_stream.flush()
        self.log_stream.flush()

    def __getattr__(self, name):
        return getattr(self.console_stream, name)


def _default_training_log_path(dataset, sa):
    dataset_name = str(dataset).strip().lower().replace(" ", "_")
    sa_name = "default" if sa is None else str(sa).strip().lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return TRAINING_LOG_DIR / f"{dataset_name}_{sa_name}_{timestamp}.log"


@contextlib.contextmanager
def _tee_training_output(log_path):
    if log_path is None:
        yield None
        return

    resolved_path = Path(log_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    current_stdout = sys.stdout
    current_stderr = sys.stderr
    with resolved_path.open("a", encoding="utf-8", buffering=1) as log_file:
        tee_stdout = _TeeStream(current_stdout, log_file)
        tee_stderr = _TeeStream(current_stderr, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(
            tee_stderr
        ):
            print(f"Training log file: {resolved_path}")
            yield resolved_path


def retrievePixels(path, height, width):
    folder_path = DATA_DIR / "Images"
    img = tf.keras.utils.load_img(str(folder_path / path), target_size=(height, width))
    return tf.keras.utils.img_to_array(img)


col = "output"
DATASET_DEFAULT_SA = {
    "scut": "gender",
    "adult": "gender",
    "german": "sex",
    "heart": "age",
    "compas": "race",
    "comm": "race",
    "lsac": "race",
}


def _binary_encode(series, positive_value):
    return series.eq(positive_value).astype(int)


def _strip_text_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    text_cols = df.select_dtypes(include="object").columns
    if len(text_cols):
        df[text_cols] = df[text_cols].apply(lambda values: values.str.strip())
    return df


def _resolve_sa_column(sa, default_sa, allowed_sa, dataset_name):
    return _resolve_sa_choice(sa, default_sa, allowed_sa, dataset_name)[1]


def _finalize_dataset(df, dependent, dataset_name, is_binary, model_df=None):
    X_train, X_test = _split_with_output(
        df if model_df is None else model_df,
        dependent,
    )
    return df, dataset_name, X_train, X_test, is_binary


def _split_with_output(df, dependent, test_size=0.2):
    X = df.drop(columns=[dependent])
    y = np.array(df[dependent])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train[col] = y_train
    X_test[col] = y_test
    return X_train, X_test


def _shared_tabular_preprocessor(is_binary, is_scut):
    if is_scut or not is_binary:
        return None
    return StandardScaler()


def _transform_tabular_frame(df, preprocessor, fit=False):
    if preprocessor is None or df is None:
        return df

    transformed = df.copy()
    feature_cols = [column for column in transformed.columns if column != col]
    transformed_values = (
        preprocessor.fit_transform(transformed[feature_cols])
        if fit
        else preprocessor.transform(transformed[feature_cols])
    )
    transformed_features = pd.DataFrame(
        np.asarray(transformed_values, dtype=np.float32),
        columns=feature_cols,
        index=transformed.index,
    )
    if col not in transformed.columns:
        return transformed_features
    return pd.concat([transformed_features, transformed[[col]]], axis=1)[
        transformed.columns
    ]


def comp_pred(test, dual_encoder):
    dataA = test["A"].tolist()
    dataB = test["B"].tolist()
    ln = len(dataA)
    predictions = []
    for i in range(ln):
        datapoint_A = np.array(dataA[i])
        datapoint_B = np.array(dataB[i])
        datapoint_A = np.expand_dims(datapoint_A, axis=0)
        datapoint_B = np.expand_dims(datapoint_B, axis=0)
        prediction = dual_encoder.predict(datapoint_A, datapoint_B)

        # prediction = round(prediction.numpy()[0][0].item()) # Labels in [0, 1]

        prediction = prediction.numpy()[0][0].item()

        # Labels: -1, 0, 1
        # if prediction < -0.33:
        #     prediction = -1
        # elif prediction > 0.33:
        #     prediction = 1
        # else:
        #     prediction = 0

        # Labels: -1, 1
        if prediction < PAIRWISE_DECISION_THRESHOLD:
            prediction = 0
        else:
            prediction = 1

        predictions.append(prediction)

    return predictions


def cal_comp(xt1, x1, xt2, x2):
    # Numerical stability: avoid 0/0 and division by zero downstream
    denom = x1 + x2
    denom = denom if denom != 0 else 1.0

    mu = (xt1 + xt2) / denom
    var = mu * (1 - mu) / denom
    return mu, var


def comparative_separation(x):
    mut11, vart11 = cal_comp(
        x["1111"],
        x["1111"] + x["0111"] + x["x111"],
        x["0011"],
        x["0011"] + x["1011"] + x["x011"],
    )
    mut00, vart00 = cal_comp(
        x["1100"],
        x["1100"] + x["0100"] + x["x100"],
        x["0000"],
        x["0000"] + x["1000"] + x["x000"],
    )
    mut10, vart10 = cal_comp(
        x["1110"],
        x["1110"] + x["0110"] + x["x110"],
        x["0001"],
        x["0001"] + x["1001"] + x["x001"],
    )
    mut01, vart01 = cal_comp(
        x["1101"],
        x["1101"] + x["0101"] + x["x101"],
        x["0010"],
        x["0010"] + x["1010"] + x["x010"],
    )

    eps = np.finfo(float).eps
    zc = (mut10 - mut01) / np.sqrt(vart10 + vart01 + eps)
    zw = (mut11 - mut00) / np.sqrt(vart11 + vart00 + eps)

    pc = norm.sf(np.abs(zc)) * 2
    pw = norm.sf(np.abs(zw)) * 2

    return [pc, pw], (mut10 - mut01), (mut11 - mut00)


# --- New: vectorized violation counting (replaces Counter + Python loops) ---
_CIJ_TO_CHAR = np.array(["0", "1", "x"], dtype=object)


def _counts_dict_from_bincount(bc: np.ndarray) -> dict:
    """
    bc length 24, encoding:
      cij_code in {0:'0',1:'1',2:'x'} (3 states)
      yij in {0,1}
      a1 in {0,1}
      a2 in {0,1}
    index = (((cij_code * 2 + yij) * 2 + a1) * 2 + a2)
    """
    out = {}
    idx = 0
    for cij_code in (0, 1, 2):
        cij_char = _CIJ_TO_CHAR[cij_code]
        for yij in (0, 1):
            for a1 in (0, 1):
                for a2 in (0, 1):
                    out[f"{cij_char}{yij}{a1}{a2}"] = int(bc[idx])
                    idx += 1
    return out


def count_violation_fast(arr: np.ndarray, i1: np.ndarray, i2: np.ndarray) -> dict:
    """
    arr columns: [c, y, a]
    i1, i2 are 1D index arrays of equal length
    Returns dict with keys like '1111', 'x011', etc. Missing keys => 0.
    """
    c1 = arr[i1, 0]
    y1 = arr[i1, 1]
    a1 = arr[i1, 2].astype(np.int8)

    c2 = arr[i2, 0]
    y2 = arr[i2, 1]
    a2 = arr[i2, 2].astype(np.int8)

    # cij_code: 2 for tie, 1 if c1>c2, 0 if c1<c2
    cij_code = np.empty_like(c1, dtype=np.int8)
    eq = c1 == c2
    gt = c1 > c2
    cij_code[eq] = 2
    cij_code[~eq & gt] = 1
    cij_code[~eq & ~gt] = 0

    yij = (y1 > y2).astype(np.int8)

    code = (((cij_code * 2 + yij) * 2 + a1) * 2 + a2).astype(np.int16)
    bc = np.bincount(code, minlength=24)

    return _counts_dict_from_bincount(bc)


def batched_score(dual_encoder, inputs, batch_size=32):
    batches = [
        dual_encoder.score(inputs[start : start + batch_size]).numpy().flatten()
        for start in range(0, len(inputs), batch_size)
    ]
    return np.concatenate(batches) if batches else np.array([])


def make_df6(n=1000, p1=0.5, p2=0.5, p3=0.5):
    df = pd.DataFrame(
        {
            "gender": (np.random.random(n) < p1).astype(int),
            "income": (np.random.random(n) < p2).astype(int),
            "pred": (np.random.random(n) < p3).astype(int),
        }
    )
    return df, "df6"


def _resolve_sa_choice(sa, default_sa, allowed_sa, dataset_name):
    allowed = {k.lower(): v for k, v in allowed_sa.items()}
    selected = default_sa.lower() if sa is None else str(sa).strip().lower()
    if selected not in allowed:
        valid = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unsupported sensitive attribute '{sa}' for dataset '{dataset_name}'. "
            f"Valid options: {valid}"
        )
    return selected, allowed[selected]


def make_scut(P="P2", sa="gender"):
    df = pd.read_csv(DATA_DIR / "ImageExp" / "Selected_Ratings.csv")
    df = df[["Filename", P]]

    # Parallel image loading with caching
    print(f"Loading {len(df)} images in parallel...")
    pixels = DataProcessing.retrievePixels_batch_vggface(df["Filename"].tolist())
    df["pixels"] = pixels
    print("Images loaded.")

    fn = df["Filename"].astype(str)
    df["gender"] = _binary_encode(fn.str[1].fillna(""), "M")
    df["race"] = _binary_encode(fn.str[0].fillna(""), "C")

    sa_col = _resolve_sa_column(
        sa,
        default_sa="gender",
        allowed_sa={"gender": "gender", "race": "race"},
        dataset_name="scut",
    )
    df = df.rename(columns={sa_col: "sa"})
    dependent = P
    model_df = df[["pixels", "sa", dependent]]
    return _finalize_dataset(
        df,
        dependent,
        f"scut_{P}",
        is_binary=False,
        model_df=model_df,
    )


def make_adult(sa="gender"):
    df = _strip_text_columns(pd.read_csv(DATA_DIR / "adult.csv", na_values=["?"]))
    df = df.dropna()
    df["gender"] = _binary_encode(df["gender"], "Male")
    df["income"] = _binary_encode(df["income"], ">50K")
    dependent = "income"

    sa_col = _resolve_sa_column(
        sa,
        default_sa="gender",
        allowed_sa={"gender": "gender", "race": "race"},
        dataset_name="adult",
    )
    if sa_col == "race":
        df["race"] = _binary_encode(df["race"], "White")

    df = df.rename(columns={sa_col: "sa"})
    dummy_cols = [
        column
        for column in ["workclass", "marital-status", "occupation", "relationship", "race"]
        if column != sa_col
    ]
    df = pd.get_dummies(
        df,
        columns=dummy_cols,
        dtype=float,
        drop_first=True,
    )

    model_df = df.drop(columns=["education", "native-country"])
    return _finalize_dataset(
        df,
        dependent,
        "adult",
        is_binary=True,
        model_df=model_df,
    )


def make_german(sa="sex"):
    df = _strip_text_columns(
        pd.read_csv(DATA_DIR / "german_credit_data.csv", index_col=0)
    ).dropna()
    df["Sex"] = df["Sex"].str.lower().eq("male").astype(int)
    df["Risk"] = df["Risk"].str.lower().eq("good").astype(int)
    dependent = "Risk"
    sa_col = _resolve_sa_column(
        sa,
        default_sa="sex",
        allowed_sa={"sex": "Sex", "age": "Age"},
        dataset_name="german",
    )
    if sa_col == "Age":
        age_median = df["Age"].median()
        df["Age"] = (df["Age"] >= age_median).astype(int)

    df = df.rename(columns={sa_col: "sa"})

    df = pd.get_dummies(
        df,
        columns=["Housing", "Saving accounts", "Checking account", "Purpose"],
        dtype=float,
        drop_first=True,
    )
    return _finalize_dataset(df, dependent, "german", is_binary=True)


def make_heart(sa="age"):
    df = _strip_text_columns(pd.read_csv(DATA_DIR / "heart.csv")).dropna()

    dependent = "output"
    sa_col = _resolve_sa_column(
        sa,
        default_sa="age",
        allowed_sa={"age": "age", "sex": "sex"},
        dataset_name="heart",
    )
    if sa_col == "age":
        df["age"] = (df["age"] >= 55).astype(int)
    df = df.rename(columns={sa_col: "sa"})
    return _finalize_dataset(df, dependent, "heart", is_binary=True)


def make_compas(sa="race"):
    features_to_keep = [
        "sex",
        "age",
        "age_cat",
        "race",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "c_charge_degree",
        "two_year_recid",
    ]
    df = _strip_text_columns(
        pd.read_csv(DATA_DIR / "compas-scores-two-years.csv", usecols=features_to_keep)
    )
    df["sex"] = _binary_encode(df["sex"], "Male")
    df["race"] = _binary_encode(df["race"], "Caucasian")
    df["two_year_recid"] = (df["two_year_recid"] == 0).astype(int)

    sa_col = _resolve_sa_column(
        sa,
        default_sa="race",
        allowed_sa={"race": "race", "sex": "sex"},
        dataset_name="compas",
    )
    df = df.rename(columns={sa_col: "sa"})

    df = pd.get_dummies(
        df, columns=["age_cat", "c_charge_degree"], dtype=float, drop_first=True
    )

    dependent = "two_year_recid"
    return _finalize_dataset(df, dependent, "compas", is_binary=True)


def make_comm(sa="race"):
    df = pd.read_csv(DATA_DIR / "communities.csv")
    df = df.fillna(0)
    race_columns = ["racepctblack", "racePctWhite", "racePctAsian", "racePctHisp"]
    majority_race = majority_pop(df[race_columns])
    df["race"] = majority_race.map(
        {"racepctblack": 0, "racePctWhite": 1, "racePctAsian": 0, "racePctHisp": 0}
    )
    df = df.drop(columns=race_columns)

    dependent = "ViolentCrimesPerPop"
    sa_col = _resolve_sa_column(
        sa,
        default_sa="race",
        allowed_sa={"race": "race"},
        dataset_name="comm",
    )

    df = df.rename(columns={sa_col: "sa"})
    return _finalize_dataset(df, dependent, "comm", is_binary=False)


def make_lsac(sa="race"):
    df = pd.read_csv(DATA_DIR / "lawschool.csv")
    df = df.dropna()

    df["race"] = (df["race"] == 7.0).astype(int)
    df["ugpa"] = df["ugpa"] / df["ugpa"].max()
    df["gender"] = df["gender"].map({"male": 1, "female": 0})
    df["bar1"] = _binary_encode(df["bar1"], "P")

    dependent = "ugpa"
    sa_col = _resolve_sa_column(
        sa,
        default_sa="race",
        allowed_sa={"race": "race", "gender": "gender"},
        dataset_name="lsac",
    )

    df = df.rename(columns={sa_col: "sa"})
    return _finalize_dataset(df, dependent, "lsac", is_binary=False)


def majority_pop(a):
    return a.apply(pd.Series.idxmax, axis=1)


def remove_outliers(data):
    """
    Performs 1D k-means after removing outliers using the IQR method.
    """
    # Reshape the data for scikit-learn
    X = data.reshape(-1, 1)

    # Calculate Q1, Q3, and IQR
    Q1, Q3 = np.percentile(X, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    X_filtered = X[(X >= lower_bound) & (X <= upper_bound)]
    return X_filtered


def kmeans_binarize_1d(data):
    data_filt = remove_outliers(data)
    km = KMeans(n_clusters=2, n_init=10, max_iter=300, random_state=42).fit(
        data_filt.reshape(-1, 1)
    )
    labels = km.predict(data.reshape(-1, 1))
    if km.cluster_centers_[0] > km.cluster_centers_[1]:
        labels = 1 - labels
    return labels


def _extract_validation_examples(pair_df, is_scut):
    if pair_df is None or len(pair_df) == 0:
        return None, None

    examples = {}
    for index_col, feature_col, target_col in (
        ("Index_A", "A", "Y_A"),
        ("Index_B", "B", "Y_B"),
    ):
        if not {index_col, feature_col, target_col}.issubset(pair_df.columns):
            return None, None
        for idx, features, target in zip(
            pair_df[index_col], pair_df[feature_col], pair_df[target_col]
        ):
            key = int(idx)
            if key in examples:
                continue
            examples[key] = (
                np.asarray(features, dtype=np.float32),
                int(target),
            )

    if not examples:
        return None, None

    ordered_examples = [examples[key] for key in sorted(examples)]
    labels = np.asarray([target for _, target in ordered_examples], dtype=np.int8)
    features = [feature for feature, _ in ordered_examples]
    if is_scut:
        return np.stack(features).astype(np.float32), labels
    return np.asarray(features, dtype=np.float32), labels


def _candidate_thresholds(scores):
    unique_scores = np.unique(np.asarray(scores, dtype=np.float64))
    if unique_scores.size == 0:
        return np.asarray([0.5], dtype=np.float64)
    if unique_scores.size == 1:
        return unique_scores.astype(np.float64)

    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    eps = max(np.finfo(np.float64).eps, np.std(unique_scores) * 1e-6)
    return np.concatenate(
        (
            [unique_scores[0] - eps],
            midpoints,
            [unique_scores[-1] + eps],
        )
    )


def _learn_binary_threshold(scores, labels, default_threshold=0.5):
    scores = np.asarray(scores, dtype=np.float64)
    labels = (np.asarray(labels) > 0).astype(np.int8)
    finite_mask = np.isfinite(scores)
    scores = scores[finite_mask]
    labels = labels[finite_mask]

    if scores.size == 0 or np.unique(labels).size < 2:
        return float(default_threshold), np.nan

    best_threshold = float(default_threshold)
    best_metric = -np.inf
    for threshold in _candidate_thresholds(scores):
        predictions = (scores >= threshold).astype(np.int8)
        metric_value = balanced_accuracy_score(labels, predictions)
        if metric_value > best_metric + 1e-12 or (
            np.isclose(metric_value, best_metric)
            and abs(threshold - default_threshold)
            < abs(best_threshold - default_threshold)
        ):
            best_metric = float(metric_value)
            best_threshold = float(threshold)

    return best_threshold, best_metric


def plot_threshold_comparison_histogram(
    raw_predictions,
    thresholded_predictions,
    threshold,
    output_path,
    title,
    bins=80,
):
    raw_predictions = np.asarray(raw_predictions, dtype=np.float64)
    finite_predictions = raw_predictions[np.isfinite(raw_predictions)]
    if finite_predictions.size == 0:
        return

    thresholded_predictions = np.asarray(thresholded_predictions, dtype=np.int8).reshape(-1)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_raw, ax_binary) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_raw.hist(
        finite_predictions,
        bins=min(bins, max(40, int(np.sqrt(finite_predictions.size) * 2))),
        color="#4C78A8",
        edgecolor="white",
        alpha=0.9,
    )
    if np.isfinite(threshold):
        ax_raw.axvline(
            threshold,
            color="#E45756",
            linestyle="--",
            linewidth=2,
            label=f"threshold={threshold:.4f}",
        )
        ax_raw.legend(frameon=False)
    ax_raw.set_title(f"{title}: before thresholding")
    ax_raw.set_xlabel("Raw prediction score")
    ax_raw.set_ylabel("Count")
    ax_raw.grid(axis="y", alpha=0.2)

    clipped_predictions = np.clip(thresholded_predictions, 0, 1)
    ax_binary.hist(
        clipped_predictions,
        bins=np.array([-0.5, 0.5, 1.5]),
        color="#72B7B2",
        edgecolor="white",
        alpha=0.9,
        rwidth=0.8,
    )
    ax_binary.set_title(f"{title}: after thresholding")
    ax_binary.set_xlabel("Thresholded prediction")
    ax_binary.set_ylabel("Count")
    ax_binary.set_xticks([0, 1])
    ax_binary.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def stats(x1, x2):
    denom = x1 + x2
    eps = np.finfo(float).eps

    # If there is no data in this slice, return a neutral mean and 0 variance
    if denom == 0:
        return 0.5, 0.0

    mu = x1 / denom
    var = mu * (1 - mu) / max(denom, eps)
    return mu, var


def separation(y, y_pred, s):
    # Robust binarization: supports continuous/regression inputs without negative
    # bincount indices while preserving 0/1 behavior for binary datasets.
    y = (np.asarray(y) > 0).astype(np.int8)
    y_pred = (np.asarray(y_pred) > 0).astype(np.int8)
    s = (np.asarray(s) > 0).astype(np.int8)

    # Encode (pred, y, s) into 0..7 and count once.
    encoded = (y_pred << 2) + (y << 1) + s
    counts = np.bincount(encoded, minlength=8)

    mut1, vart1 = stats(counts[7], counts[3])  # 111 vs 011
    mut0, vart0 = stats(counts[6], counts[2])  # 110 vs 010

    eps = np.finfo(float).eps
    zt = (mut1 - mut0) / np.sqrt(max(vart1 + vart0, eps))
    pt = norm.sf(np.abs(zt)) * 2

    muf1, varf1 = stats(counts[5], counts[1])  # 101 vs 001
    muf0, varf0 = stats(counts[4], counts[0])  # 100 vs 000
    zf = (muf1 - muf0) / np.sqrt(max(varf1 + varf0, eps))
    pf = norm.sf(np.abs(zf)) * 2

    return [pt, pf]


alpha = 0.05
r = 100
FAIRNESS_LAMBDA = 0.1


def _build_train_pairs(
    train_vals, col_idx, sa_idx, pixels_idx, is_scut, use_all_pairs, num_comp_train
):
    keep_cols = np.ones(train_vals.shape[1], dtype=bool)
    keep_cols[col_idx] = False

    if use_all_pairs:
        num_train = len(train_vals)
        idx_a, idx_b = np.where(~np.eye(num_train, dtype=bool))
        labels = np.sign(
            train_vals[idx_a, col_idx] - train_vals[idx_b, col_idx]
        ).astype(int)
        valid_mask = labels != 0
        idx_a = idx_a[valid_mask]
        idx_b = idx_b[valid_mask]
        rows_a, rows_b, labels = (
            train_vals[idx_a],
            train_vals[idx_b],
            labels[valid_mask],
        )
    else:
        n_train = len(train_vals)
        all_idx_a = []
        all_idx_b = []
        used_pairs = set()

        for idx_a in range(n_train):
            candidates = [
                idx_b
                for idx_b in range(n_train)
                if idx_b != idx_a
                and (min(idx_a, idx_b), max(idx_a, idx_b)) not in used_pairs
            ]
            if not candidates:
                continue

            k = min(num_comp_train, len(candidates))
            partners = np.random.choice(candidates, size=k, replace=False)
            for idx_b in partners:
                pair = (min(idx_a, int(idx_b)), max(idx_a, int(idx_b)))
                used_pairs.add(pair)
                all_idx_a.extend([idx_a, int(idx_b)])
                all_idx_b.extend([int(idx_b), idx_a])

        rows_a = train_vals[np.array(all_idx_a)]
        rows_b = train_vals[np.array(all_idx_b)]
        diffs = rows_a[:, col_idx] - rows_b[:, col_idx]
        valid_mask = diffs != 0
        idx_a = np.asarray(all_idx_a, dtype=np.int64)[valid_mask]
        idx_b = np.asarray(all_idx_b, dtype=np.int64)[valid_mask]
        rows_a = rows_a[valid_mask]
        rows_b = rows_b[valid_mask]
        labels = np.where(diffs[valid_mask] > 0, 1, -1)

    if is_scut:
        feats_a = np.stack(rows_a[:, pixels_idx]).astype(np.float32)
        feats_b = np.stack(rows_b[:, pixels_idx]).astype(np.float32)
        train_pairs = [
            {
                "Index_A": int(ia),
                "Index_B": int(ib),
                "A": fa,
                "B": fb,
                "Label": int(lbl),
                "Y_A": float(ra[col_idx]),
                "Y_B": float(rb[col_idx]),
                "SA_A": int(ra[sa_idx]),
                "SA_B": int(rb[sa_idx]),
            }
            for ia, ib, fa, fb, lbl, ra, rb in zip(
                idx_a,
                idx_b,
                feats_a,
                feats_b,
                labels,
                rows_a,
                rows_b,
            )
        ]
    else:
        feats_a = rows_a[:, keep_cols]
        feats_b = rows_b[:, keep_cols]
        train_pairs = [
            {
                "Index_A": int(ia),
                "Index_B": int(ib),
                "A": fa.tolist(),
                "B": fb.tolist(),
                "Label": int(lbl),
                "Y_A": float(ra[col_idx]),
                "Y_B": float(rb[col_idx]),
                "SA_A": int(ra[sa_idx]),
                "SA_B": int(rb[sa_idx]),
            }
            for ia, ib, fa, fb, lbl, ra, rb in zip(
                idx_a,
                idx_b,
                feats_a,
                feats_b,
                labels,
                rows_a,
                rows_b,
            )
        ]

    pair_meta = [
        {"AB": (ra[sa_idx], rb[sa_idx]), "AY": ((ra[sa_idx], rb[sa_idx]), int(lbl))}
        for ra, rb, lbl in zip(rows_a, rows_b, labels)
    ]
    return train_pairs, pair_meta


def _compute_pair_weights(pair_meta_df):
    ab_array = np.array(pair_meta_df["AB"].tolist())
    is_same_group = ab_array[:, 0] == ab_array[:, 1]
    p_aij = pair_meta_df["AB"].value_counts(normalize=True)
    p_aij_yij = pair_meta_df["AY"].value_counts(normalize=True)

    weights = np.ones(len(pair_meta_df))
    diff_group_mask = ~is_same_group
    weights[diff_group_mask] = (
        pair_meta_df.loc[diff_group_mask, "AB"].map(p_aij)
        / (2 * pair_meta_df.loc[diff_group_mask, "AY"].map(p_aij_yij))
    ).values
    return weights


def _validation_size(length, val_fraction, min_val_size=1):
    if length < 2 or val_fraction <= 0:
        return 0
    return min(length - 1, max(min_val_size, int(np.ceil(length * val_fraction))))


def _split_validation_frame(df, val_fraction=0.1, stratify_col=None, min_val_size=1):
    if df is None:
        return df, None

    val_size = _validation_size(len(df), val_fraction, min_val_size)
    if val_size == 0:
        return df, None

    stratify = None
    if stratify_col is not None and stratify_col in df.columns:
        counts = df[stratify_col].value_counts()
        if len(counts) > 1 and (counts >= 2).all():
            stratify = df[stratify_col]

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=stratify,
    )
    return train_df, val_df


def _split_validation_array(features, targets, val_fraction=0.1, min_val_size=1):
    val_size = _validation_size(len(features), val_fraction, min_val_size)
    if val_size == 0:
        return features, None, targets, None

    train_x, val_x, train_y, val_y = train_test_split(
        features,
        targets,
        test_size=val_size,
    )
    return train_x, val_x, train_y, val_y


def _load_dataset(dataset_name, sa=None):
    dataset_loaders = {
        "scut": make_scut,
        "adult": make_adult,
        "german": make_german,
        "heart": make_heart,
        "compas": make_compas,
        "comm": make_comm,
        "lsac": make_lsac,
    }
    if dataset_name not in dataset_loaders:
        valid = ", ".join(sorted(dataset_loaders))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Valid options: {valid}")
    return dataset_loaders[dataset_name](sa=sa)


def _prepare_encoder_splits(
    train, is_scut, use_all_pairs, num_comp_train, validation_fraction
):
    train_vals = train.values
    train_cols = list(train.columns)
    col_idx, sa_idx = train_cols.index(col), train_cols.index("sa")
    pixels_idx = train_cols.index("pixels") if is_scut else None

    train_pairs, pair_meta = _build_train_pairs(
        train_vals=train_vals,
        col_idx=col_idx,
        sa_idx=sa_idx,
        pixels_idx=pixels_idx,
        is_scut=is_scut,
        use_all_pairs=use_all_pairs,
        num_comp_train=num_comp_train,
    )
    if not train_pairs:
        raise ValueError("No comparable training pairs were generated.")

    total_pairs = len(train_pairs)
    encoder_train = pd.DataFrame(train_pairs)
    weights = _compute_pair_weights(pd.DataFrame(pair_meta))
    encoder_train, encoder_val = _split_validation_frame(
        encoder_train,
        val_fraction=validation_fraction,
        stratify_col="Label",
    )
    train_weights = weights[encoder_train.index.to_numpy()]
    encoder_train = encoder_train.reset_index(drop=True)

    if encoder_val is None:
        return encoder_train, None, train_weights, None, total_pairs

    val_weights = weights[encoder_val.index.to_numpy()]
    encoder_val = encoder_val.reset_index(drop=True)
    return encoder_train, encoder_val, train_weights, val_weights, total_pairs


def _train_shared_models(
    run_idx,
    num_runs,
    train_df,
    val_df,
    train_weights,
    val_weights,
    df_name,
    epochs,
    train_fairreg,
    tabular_encoder_type,
    is_binary,
):
    activation = "sigmoid" if is_binary else "linear"
    preprocessing_note = ", baseline preprocessing" if is_binary and "scut" not in df_name.lower() else ""
    base_kwargs = {
        "train": train_df,
        "val": val_df,
        "y_true": train_df["Label"].tolist(),
        "shared": True,
        "epochs": epochs,
        "df_name": df_name,
        "tabular_encoder_type": tabular_encoder_type,
        "output_activation": activation,
    }
    model_specs = {
        "unweight": ("unweighted", {}),
        "weighted": (
            "weighted",
            {"train_weights": train_weights, "val_weights": val_weights},
        ),
    }
    if train_fairreg:
        model_specs["fairreg"] = (
            "fairreg",
            {"fairness_lambda": FAIRNESS_LAMBDA},
        )

    models = {}
    for name, (label, extra_kwargs) in model_specs.items():
        print(
            f"[Run {run_idx + 1}/{num_runs}] Training shared encoder "
            f"({label}, {activation} head{preprocessing_note})..."
        )
        models[name] = Classification.train_model(**base_kwargs, **extra_kwargs)
    return models


def _train_optional_baselines(
    run_idx,
    num_runs,
    train,
    is_scut,
    is_binary,
    train_single_encoder,
    validation_fraction,
    epochs,
    tabular_encoder_type,
):
    baselines = {}

    if is_scut:
        print(f"[Run {run_idx + 1}/{num_runs}] Training SCUT VGG-Face baseline...")
        train_pixels = np.stack(train["pixels"].values).astype(np.float32)
        train_targets = train[col].values.astype(np.float32)
        train_pixels, val_pixels, train_targets, val_targets = _split_validation_array(
            train_pixels,
            train_targets,
            val_fraction=validation_fraction,
        )
        baselines["vgg_baseline"] = Classification.train_scut_vggface_baseline(
            train_pixels=train_pixels,
            y_train=train_targets,
            val_pixels=val_pixels,
            y_val=val_targets,
            epochs=epochs,
            batch_size=2,
        )
        return baselines

    if train_single_encoder:
        activation = "sigmoid" if is_binary else "linear"
        print(
            f"[Run {run_idx + 1}/{num_runs}] Training single encoder baseline "
            f"(baseline preprocessing, {tabular_encoder_type} structure, "
            f"{activation} head)..."
        )
        train_features = train.drop(columns=[col]).values.astype(np.float32)
        train_targets = train[col].values.astype(np.float32)
        (
            train_features,
            val_features,
            train_targets,
            val_targets,
        ) = _split_validation_array(
            train_features,
            train_targets,
            val_fraction=validation_fraction,
        )
        baselines["single_encoder"] = Classification.train_single_encoder_baseline(
            train_features=train_features,
            y_train=train_targets,
            is_binary=is_binary,
            output_activation=activation,
            val_features=val_features,
            y_val=val_targets,
            epochs=epochs,
            tabular_encoder_type=tabular_encoder_type,
        )

    return baselines


def _shared_model_inputs(test_features, is_scut):
    if is_scut:
        return np.stack(test_features["pixels"].values).astype(np.float32)
    return test_features.values


def _score_shared_model(model, test_inputs, is_scut):
    if is_scut:
        return batched_score(model, test_inputs, batch_size=4)
    return model.score(test_inputs).numpy().flatten()


def _plot_binary_histograms(
    shared_predictions,
    thresholded_predictions,
    shared_thresholds,
    df_name,
    sa_name,
    run_idx,
    pair_strategy,
):
    hist_prefix = f"{df_name}_{sa_name}_run{run_idx + 1}_{pair_strategy}"
    hist_specs = {
        "unweight": ("unweighted", f"{df_name} unweighted predictions"),
        "weighted": ("weighted", f"{df_name} weighted predictions"),
        "fairreg": ("fairreg", f"{df_name} fairreg predictions"),
    }
    for name, predictions in shared_predictions.items():
        suffix, title = hist_specs[name]
        plot_threshold_comparison_histogram(
            raw_predictions=predictions,
            thresholded_predictions=thresholded_predictions[name],
            threshold=shared_thresholds[name],
            output_path=HISTOGRAM_DIR / f"{hist_prefix}_{suffix}.png",
            title=f"{title} (run {run_idx + 1})",
        )


def _sample_comparable_pairs(y_values, num_pairs):
    i1 = np.empty(num_pairs, dtype=np.int64)
    i2 = np.empty(num_pairs, dtype=np.int64)
    filled = 0

    while filled < num_pairs:
        remaining = num_pairs - filled
        batch_size = max(remaining * 2, 256)
        idx1 = np.random.randint(0, len(y_values), size=batch_size)
        idx2 = np.random.randint(0, len(y_values), size=batch_size)
        valid = y_values[idx1] != y_values[idx2]
        if not np.any(valid):
            continue

        take = min(remaining, int(valid.sum()))
        i1[filled : filled + take] = idx1[valid][:take]
        i2[filled : filled + take] = idx2[valid][:take]
        filled += take

    return i1, i2


def _compute_violation_rates(eval_arrays, is_binary, num_comp_pairs_ratio):
    first_array = next(iter(eval_arrays.values()))
    n_rows = len(first_array)
    y_values = first_array[:, 1]
    num_comp_pairs_eval = max(1, int(np.ceil(float(num_comp_pairs_ratio) * n_rows)))

    separation_rates = {name: np.nan for name in eval_arrays}
    if is_binary:
        separation_counts = {name: 0 for name in eval_arrays}
        sample_size = max(1, n_rows // 2)
        for _ in range(r):
            selected = np.random.choice(n_rows, size=sample_size, replace=False)
            for name, data in eval_arrays.items():
                separation_counts[name] += (
                    min(separation(data[selected, 1], data[selected, 0], data[selected, 2]))
                    < alpha
                )
        separation_rates = {
            name: count / r for name, count in separation_counts.items()
        }

    comparative_counts = {name: 0 for name in eval_arrays}
    if np.unique(y_values).size > 1:
        for _ in range(r):
            i1, i2 = _sample_comparable_pairs(y_values, num_comp_pairs_eval)
            for name, data in eval_arrays.items():
                comparative_counts[name] += (
                    min(comparative_separation(count_violation_fast(data, i1, i2))[0])
                    < alpha
                )

    comparative_rates = {name: count / r for name, count in comparative_counts.items()}
    return separation_rates, comparative_rates, num_comp_pairs_eval


def _reference_results(y_true, predictions, sa_values, is_binary):
    if predictions is None:
        if is_binary:
            return {
                "Acc_lr": np.nan,
                "F1_lr": np.nan,
                "AOD_lr": np.nan,
                "EOD_lr": np.nan,
                "I_sep_lr": np.nan,
            }
        return {
            "MSE_lr": np.nan,
            "spearman_lr": np.nan,
            "pearson_lr": np.nan,
            "I_sep_lr": np.nan,
        }

    metrics = Metrics(y_true, predictions)
    result = {"I_sep_lr": metrics.MI_con_info(sa_values)}
    if is_binary:
        result.update(
            {
                "Acc_lr": accuracy_score(y_true, predictions),
                "F1_lr": f1_score(y_true, predictions),
                "AOD_lr": metrics.AOD(sa_values),
                "EOD_lr": metrics.EOD(sa_values),
            }
        )
    else:
        result.update(
            {
                "MSE_lr": metrics.mse(),
                "spearman_lr": metrics.spearmanr_coefficient(),
                "pearson_lr": metrics.pearsonr_coefficient(),
            }
        )
    return result


def _prediction_value(predictions_by_name, name, scorer, y_true):
    predictions = predictions_by_name.get(name)
    return scorer(y_true, predictions) if predictions is not None else np.nan


def _metric_value(metrics_by_name, name, method_name, *args):
    metrics = metrics_by_name.get(name)
    return getattr(metrics, method_name)(*args) if metrics is not None else np.nan


def _rate_value(values_by_name, name):
    return values_by_name.get(name, np.nan)


def _run_experiments_impl(
    num_runs=10,
    dataset="scut",
    sa=None,
    use_all_pairs=False,
    num_comp_train=1,
    train_fairreg=True,
    train_single_encoder=True,
    plot_histograms=True,
    num_comp_pairs_ratio=0.1,
    model_epochs=100,
    validation_fraction=0.1,
    tabular_encoder_type="cnn",
):
    results = []
    output_df_name = None
    output_sa_name = (
        DATASET_DEFAULT_SA.get(dataset, "sa") if sa is None else str(sa).strip().lower()
    )
    effective_use_all_pairs = use_all_pairs or dataset == "heart"
    pair_strategy = "all" if effective_use_all_pairs else str(num_comp_train)
    output_nc = 0

    for run_idx in range(num_runs):
        _, df_name, train, test, is_binary = _load_dataset(dataset, sa=output_sa_name)
        output_df_name = df_name
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        is_scut = "scut" in df_name
        shared_preprocessor = _shared_tabular_preprocessor(
            is_binary=is_binary,
            is_scut=is_scut,
        )
        shared_train = _transform_tabular_frame(
            train,
            shared_preprocessor,
            fit=True,
        )
        shared_test = _transform_tabular_frame(
            test,
            shared_preprocessor,
        )
        y_train = train[col].values
        y_test = test[col].values
        test_features = test.drop(columns=[col])
        sa_values = test_features["sa"].values
        train_features = train.drop(columns=[col])
        shared_test_features = shared_test.drop(columns=[col])
        test_inputs = _shared_model_inputs(shared_test_features, is_scut)

        (
            data_tr_encoder,
            data_val_encoder,
            train_weights,
            val_weights,
            output_nc,
        ) = _prepare_encoder_splits(
            shared_train,
            is_scut=is_scut,
            use_all_pairs=effective_use_all_pairs,
            num_comp_train=num_comp_train,
            validation_fraction=validation_fraction,
        )

        shared_models = _train_shared_models(
            run_idx=run_idx,
            num_runs=num_runs,
            train_df=data_tr_encoder,
            val_df=data_val_encoder,
            train_weights=train_weights,
            val_weights=val_weights,
            df_name=df_name,
            epochs=model_epochs,
            train_fairreg=train_fairreg,
            tabular_encoder_type=tabular_encoder_type,
            is_binary=is_binary,
        )
        optional_baselines = _train_optional_baselines(
            run_idx=run_idx,
            num_runs=num_runs,
            train=train,
            is_scut=is_scut,
            is_binary=is_binary,
            train_single_encoder=train_single_encoder,
            validation_fraction=validation_fraction,
            epochs=model_epochs,
            tabular_encoder_type=tabular_encoder_type,
        )

        if is_scut:
            reference_predictions = None
        elif is_binary:
            reference_predictions = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000),
            ).fit(train_features, y_train).predict(test_features)
        else:
            reference_predictions = LinearRegression().fit(
                train_features, y_train
            ).predict(test_features)

        shared_predictions = {
            name: _score_shared_model(model, test_inputs, is_scut)
            for name, model in shared_models.items()
        }
        shared_thresholds = {}
        if is_binary:
            val_features, val_targets = _extract_validation_examples(
                data_val_encoder,
                is_scut=is_scut,
            )
            for name, model in shared_models.items():
                if val_features is None:
                    threshold, val_metric = 0.5, np.nan
                else:
                    val_scores = _score_shared_model(model, val_features, is_scut)
                    threshold, val_metric = _learn_binary_threshold(
                        val_scores,
                        val_targets,
                        default_threshold=0.5,
                    )
                shared_thresholds[name] = threshold
                metric_display = (
                    "n/a" if np.isnan(val_metric) else f"{val_metric:.4f}"
                )
                print(
                    f"[Run {run_idx + 1}/{num_runs}] Shared encoder ({name}) "
                    f"validation threshold={threshold:.4f} "
                    f"(balanced_accuracy={metric_display})."
                )
        final_predictions = (
            {
                name: (predictions >= shared_thresholds[name]).astype(int)
                for name, predictions in shared_predictions.items()
            }
            if is_binary
            else dict(shared_predictions)
        )
        if is_binary and plot_histograms:
            _plot_binary_histograms(
                shared_predictions=shared_predictions,
                thresholded_predictions=final_predictions,
                shared_thresholds=shared_thresholds,
                df_name=df_name,
                sa_name=output_sa_name,
                run_idx=run_idx,
                pair_strategy=pair_strategy,
            )

        if "single_encoder" in optional_baselines:
            single_encoder_predictions = Classification.predict_single_encoder_baseline(
                optional_baselines["single_encoder"],
                test_features.values.astype(np.float32),
            )
            final_predictions["single_encoder"] = (
                (single_encoder_predictions >= 0.5).astype(int)
                if is_binary
                else single_encoder_predictions
            )
        if "vgg_baseline" in optional_baselines:
            final_predictions["vgg_baseline"] = (
                Classification.predict_scut_vggface_baseline(
                    optional_baselines["vgg_baseline"],
                    test_inputs,
                    batch_size=4,
                )
            )

        eval_arrays = {
            name: np.column_stack([predictions, y_test, sa_values])
            for name, predictions in final_predictions.items()
        }
        violation_rates, comparative_rates, num_comp_pairs_eval = (
            _compute_violation_rates(
                eval_arrays,
                is_binary=is_binary,
                num_comp_pairs_ratio=num_comp_pairs_ratio,
            )
        )
        method_metrics = {
            name: Metrics(y_test, predictions)
            for name, predictions in final_predictions.items()
        }
        reference_results = _reference_results(
            y_test,
            reference_predictions,
            sa_values,
            is_binary=is_binary,
        )

        if is_binary:
            result = {
                "fairness_lambda": FAIRNESS_LAMBDA,
                "num_comp_pairs_eval": num_comp_pairs_eval,
                **reference_results,
                "Acc_single_encoder": _prediction_value(
                    final_predictions, "single_encoder", accuracy_score, y_test
                ),
                "Acc_unweight": _prediction_value(
                    final_predictions, "unweight", accuracy_score, y_test
                ),
                "Acc_weighted": _prediction_value(
                    final_predictions, "weighted", accuracy_score, y_test
                ),
                "Acc_fairreg": _prediction_value(
                    final_predictions, "fairreg", accuracy_score, y_test
                ),
                "F1_single_encoder": _prediction_value(
                    final_predictions, "single_encoder", f1_score, y_test
                ),
                "F1_unweight": _prediction_value(
                    final_predictions, "unweight", f1_score, y_test
                ),
                "F1_weighted": _prediction_value(
                    final_predictions, "weighted", f1_score, y_test
                ),
                "F1_fairreg": _prediction_value(
                    final_predictions, "fairreg", f1_score, y_test
                ),
                "AOD_single_encoder": _metric_value(
                    method_metrics, "single_encoder", "AOD", sa_values
                ),
                "AOD_unweight": _metric_value(
                    method_metrics, "unweight", "AOD", sa_values
                ),
                "AOD_weighted": _metric_value(
                    method_metrics, "weighted", "AOD", sa_values
                ),
                "AOD_fairreg": _metric_value(
                    method_metrics, "fairreg", "AOD", sa_values
                ),
                "EOD_single_encoder": _metric_value(
                    method_metrics, "single_encoder", "EOD", sa_values
                ),
                "EOD_unweight": _metric_value(
                    method_metrics, "unweight", "EOD", sa_values
                ),
                "EOD_weighted": _metric_value(
                    method_metrics, "weighted", "EOD", sa_values
                ),
                "EOD_fairreg": _metric_value(
                    method_metrics, "fairreg", "EOD", sa_values
                ),
                "I_sep_single_encoder_bi": _metric_value(
                    method_metrics, "single_encoder", "MI_con_info", sa_values
                ),
                "I_sep_bi": _metric_value(
                    method_metrics, "unweight", "MI_con_info", sa_values
                ),
                "I_sep_weighted_bi": _metric_value(
                    method_metrics, "weighted", "MI_con_info", sa_values
                ),
                "I_sep_fairreg_bi": _metric_value(
                    method_metrics, "fairreg", "MI_con_info", sa_values
                ),
                "violate_r": _rate_value(violation_rates, "unweight"),
                "violate_r_single_encoder": _rate_value(
                    violation_rates, "single_encoder"
                ),
                "violate_r_weighted": _rate_value(violation_rates, "weighted"),
                "violate_r_fairreg": _rate_value(violation_rates, "fairreg"),
                "violate_comp_r": _rate_value(comparative_rates, "unweight"),
                "violate_comp_r_single_encoder": _rate_value(
                    comparative_rates, "single_encoder"
                ),
                "violate_comp_r_w": _rate_value(comparative_rates, "weighted"),
                "violate_comp_r_fairreg": _rate_value(
                    comparative_rates, "fairreg"
                ),
            }
        else:
            result = {
                "fairness_lambda": FAIRNESS_LAMBDA,
                "num_comp_pairs_eval": num_comp_pairs_eval,
                **reference_results,
                "MSE_single_encoder": _metric_value(
                    method_metrics, "single_encoder", "mse"
                ),
                "MSE_unweight": _metric_value(method_metrics, "unweight", "mse"),
                "MSE_weight": _metric_value(method_metrics, "weighted", "mse"),
                "MSE_vgg_baseline": _metric_value(
                    method_metrics, "vgg_baseline", "mse"
                ),
                "MSE_fairreg": _metric_value(method_metrics, "fairreg", "mse"),
                "spearman_single_encoder": _metric_value(
                    method_metrics, "single_encoder", "spearmanr_coefficient"
                ),
                "spearman_unweighted": _metric_value(
                    method_metrics, "unweight", "spearmanr_coefficient"
                ),
                "spearman_weighted": _metric_value(
                    method_metrics, "weighted", "spearmanr_coefficient"
                ),
                "spearman_vgg_baseline": _metric_value(
                    method_metrics, "vgg_baseline", "spearmanr_coefficient"
                ),
                "spearman_fairreg": _metric_value(
                    method_metrics, "fairreg", "spearmanr_coefficient"
                ),
                "pearson_single_encoder": _metric_value(
                    method_metrics, "single_encoder", "pearsonr_coefficient"
                ),
                "pearson_unweighted": _metric_value(
                    method_metrics, "unweight", "pearsonr_coefficient"
                ),
                "pearson_weighted": _metric_value(
                    method_metrics, "weighted", "pearsonr_coefficient"
                ),
                "pearson_vgg_baseline": _metric_value(
                    method_metrics, "vgg_baseline", "pearsonr_coefficient"
                ),
                "pearson_fairreg": _metric_value(
                    method_metrics, "fairreg", "pearsonr_coefficient"
                ),
                "I_sep_single_encoder_bi": _metric_value(
                    method_metrics, "single_encoder", "MI_con_info", sa_values
                ),
                "I_sep_bi": _metric_value(
                    method_metrics, "unweight", "MI_con_info", sa_values
                ),
                "I_sep_weighted_bi": _metric_value(
                    method_metrics, "weighted", "MI_con_info", sa_values
                ),
                "I_sep_vgg_baseline_bi": _metric_value(
                    method_metrics, "vgg_baseline", "MI_con_info", sa_values
                ),
                "I_sep_fairreg_bi": _metric_value(
                    method_metrics, "fairreg", "MI_con_info", sa_values
                ),
                "violate_r": _rate_value(violation_rates, "unweight"),
                "violate_r_single_encoder": _rate_value(
                    violation_rates, "single_encoder"
                ),
                "violate_r_weighted": _rate_value(violation_rates, "weighted"),
                "violate_r_vgg_baseline": _rate_value(
                    violation_rates, "vgg_baseline"
                ),
                "violate_r_fairreg": _rate_value(violation_rates, "fairreg"),
                "violate_comp_r": _rate_value(comparative_rates, "unweight"),
                "violate_comp_r_single_encoder": _rate_value(
                    comparative_rates, "single_encoder"
                ),
                "violate_comp_r_w": _rate_value(comparative_rates, "weighted"),
                "violate_comp_r_vgg_baseline": _rate_value(
                    comparative_rates, "vgg_baseline"
                ),
                "violate_comp_r_fairreg": _rate_value(
                    comparative_rates, "fairreg"
                ),
            }
        results.append(result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(
        RESULTS_DIR
        / f"{output_df_name}_{output_sa_name}_{output_nc}_{pair_strategy}.csv",
        index=False,
    )


def run_experiments(
    num_runs=10,
    dataset="scut",
    sa=None,
    use_all_pairs=False,
    num_comp_train=1,
    train_fairreg=True,
    train_single_encoder=True,
    plot_histograms=True,
    num_comp_pairs_ratio=0.1,
    model_epochs=100,
    validation_fraction=0.1,
    tabular_encoder_type="cnn",
    training_log_path=None,
):
    with _tee_training_output(training_log_path) as resolved_log_path:
        if resolved_log_path is not None:
            print(
                "Starting experiments "
                f"(dataset={dataset}, sensitive_attribute={sa or 'default'})."
            )
        _run_experiments_impl(
            num_runs=num_runs,
            dataset=dataset,
            sa=sa,
            use_all_pairs=use_all_pairs,
            num_comp_train=num_comp_train,
            train_fairreg=train_fairreg,
            train_single_encoder=train_single_encoder,
            plot_histograms=plot_histograms,
            num_comp_pairs_ratio=num_comp_pairs_ratio,
            model_epochs=model_epochs,
            validation_fraction=validation_fraction,
            tabular_encoder_type=tabular_encoder_type,
        )


if __name__ == "__main__":
    DATASET = "compas"  # scut, adult, german, heart, compas, comm, lsac
    SA = None  # None uses dataset default; e.g. "race", "sex", "gender", "age"
    NUM_RUNS = 5
    USE_ALL_PAIRS = False  # Set False to use a fixed number of training pairs per instance.
    NUM_COMP_TRAIN = 5
    TRAIN_FAIRREG = False  # Set False to disable FairReg model training.
    TRAIN_SINGLE_ENCODER = True  # Set False to skip single-encoder baseline training.
    PLOT_HISTOGRAMS = True  # Set False to skip writing prediction histogram images.
    NUM_COMP_PAIRS_RATIO = 0.1
    MODEL_EPOCHS = 500
    TABULAR_ENCODER_TYPE = "linear"  # Options for tabular datasets: "cnn", "linear"
    TRAINING_LOG_PATH = _default_training_log_path(DATASET, SA)

    run_experiments(
        num_runs=NUM_RUNS,
        dataset=DATASET,
        sa=SA,
        use_all_pairs=USE_ALL_PAIRS,
        num_comp_train=NUM_COMP_TRAIN,
        train_fairreg=TRAIN_FAIRREG,
        train_single_encoder=TRAIN_SINGLE_ENCODER,
        plot_histograms=PLOT_HISTOGRAMS,
        num_comp_pairs_ratio=NUM_COMP_PAIRS_RATIO,
        model_epochs=MODEL_EPOCHS,
        tabular_encoder_type=TABULAR_ENCODER_TYPE,
        training_log_path=TRAINING_LOG_PATH,
    )

    # TODO: Changing test size to 10% of testing pairs, change training /testing split to 90/10, and rerunning experiments.
    # TODO: Add pearson and spearman metrics for scut dataset.
    # TODO: try sigmoid activation encoder for scut
    
    # TODO: Add contribution paragraph

    # TODO: Try using sigmoid activation for non-scut datasets as well, to see if it improves fairness metrics. Plot the prediction and see if it's already well-seperated
    # TODO: Switch to SGD compiler and try different learning rates, including decaying learning rates.

    # TODO: Include another baseline with one encoder
    # TODO: Try sigmoid with single encoder
    # TODO: Try SGD with non-scut datasets as well, to see if it improves faßirness metrics
    # TODO: Run on local and see acuuracy change

    # TODO: Try linear single encoder with regression dataset, and sigmoid single encoder with classification datasets

    # TODO: Try simpler encoder architectures for tabular datasets, to see if it improves fairness metrics. Maybe start with 1-2 hidden layers and smaller hidden sizes.

    # TODO: Adult data is not performing well, and single encoder model is not showing the same result with logistic regression model.
