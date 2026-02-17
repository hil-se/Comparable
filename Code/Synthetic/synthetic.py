import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Suppress TensorFlow C++ INFO/WARNING logs (e.g., local_rendezvous messages).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

import Classification
import DataProcessing
from metrics import Metrics

isBinary = True
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "Data"
RESULTS_DIR = BASE_DIR / "Results"


def retrievePixels(path, height, width):
    # img = tf.keras.utils.load_img("../data/images/"+path, grayscale=False
    folder_path = DATA_DIR / "Images"
    # folder_path = "../../../XAI_Image/data/images/"
    img = tf.keras.utils.load_img(str(folder_path / path), target_size=(height, width))
    x = tf.keras.utils.img_to_array(img)
    return x


col = "output"


def _split_with_output(df, dependent, test_size=0.5):
    X = df.drop(columns=[dependent])
    y = np.array(df[dependent])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train[col] = y_train
    X_test[col] = y_test
    return X_train, X_test


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
        if prediction < 0:
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
    n = len(inputs)
    scores = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_scores = dual_encoder.score(inputs[start:end]).numpy().flatten()
        scores.append(batch_scores)
    return np.concatenate(scores) if scores else np.array([])


def make_df6(n=1000, p1=0.5, p2=0.5, p3=0.5):
    # n is the number of data points.
    # 0 <= p <= 1 is the sampling probability of Male (sex=1).
    keys = ["gender", "income", "pred"]
    data = {key: [] for key in keys}
    for i in range(n):
        rand1 = np.random.random()
        rand2 = np.random.random()
        rand3 = np.random.random()
        gender = 1 if rand1 < p1 else 0
        income = 1 if rand2 < p2 else 0
        pred = 1 if rand3 < p3 else 0
        data["gender"].append(gender)
        data["income"].append(income)
        data["pred"].append(pred)
    df = pd.DataFrame(data, columns=keys)
    return df, "df6"


def make_scut(P="P1"):
    df = pd.read_csv(DATA_DIR / "ImageExp" / "Selected_Ratings.csv")
    df = df[["Filename", P]]

    # Parallel image loading with caching
    print(f"Loading {len(df)} images in parallel...")
    pixels = DataProcessing.retrievePixels_batch(df["Filename"].tolist())
    df["pixels"] = [p / 255.0 for p in pixels]
    print("Images loaded.")

    fn = df["Filename"].astype(str)
    df["race"] = fn.str[0].fillna("")
    df["gender"] = fn.str[1].fillna("")

    df["gender"] = df["gender"].apply(lambda x: 1 if x == "M" else 0)
    df["race"] = df["race"].apply(lambda x: 1 if x == "C" else 0)

    sa = "gender"

    df = df.rename(columns={sa: "sa"})

    global isBinary
    dependent = P

    model_df = df[["pixels", "sa", dependent]]
    X_train, X_test = _split_with_output(model_df, dependent)

    isBinary = False

    return df, "scut" + "_" + str(P), X_train, X_test


def make_adult():
    # seed = 18
    df = pd.read_csv(DATA_DIR / "adult.csv", na_values=["?"])
    # df = df.sample(frac=0.1)
    df = df.dropna()
    df["gender"] = df["gender"].apply(lambda x: 1 if x == "Male" else 0)
    df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)
    dependent = "income"

    global isBinary
    sa = "gender"

    df = df.rename(columns={sa: "sa"})

    df = pd.get_dummies(
        df,
        columns=["workclass", "marital-status", "occupation", "relationship", "race"],
        dtype=float,
        drop_first=True,
    )

    model_df = df.drop(columns=["education", "native-country"])
    X_train, X_test = _split_with_output(model_df, dependent)

    isBinary = True

    return df, "adult", X_train, X_test


def make_german():
    # seed = 42
    df = pd.read_csv(DATA_DIR / "german_credit_data.csv", index_col=0)
    df = df.dropna()
    df["Sex"] = df["Sex"].apply(lambda x: 1 if x == "male" else 0)
    df["Risk"] = df["Risk"].apply(lambda x: 1 if x == "good" else 0)

    global isBinary
    dependent = "Risk"
    sa = "Sex"

    df = df.rename(columns={sa: "sa"})

    df = pd.get_dummies(
        df,
        columns=["Housing", "Saving accounts", "Checking account", "Purpose"],
        dtype=float,
        drop_first=True,
    )

    X_train, X_test = _split_with_output(df, dependent)

    isBinary = True

    return df, "german", X_train, X_test


def make_heart():
    # seed = 42
    df = pd.read_csv(DATA_DIR / "heart.csv")
    df = df.dropna()

    global isBinary
    dependent = "output"
    sa = "sex"

    df = df.rename(columns={sa: "sa"})

    X_train, X_test = _split_with_output(df, dependent)

    isBinary = True

    return df, "heart", X_train, X_test


def make_compas():
    df = pd.read_csv(DATA_DIR / "compas-scores-two-years.csv")
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
    df = df[features_to_keep]
    df["sex"] = df["sex"].apply(lambda x: 1 if x == "Male" else 0)
    # discretize race: Caucasian vs. non-Caucasian
    df["race"] = df["race"].apply(lambda x: 1 if x == "Caucasian" else 0)
    # prefer 0 (no recid) as label 1
    df["two_year_recid"] = df["two_year_recid"].apply(lambda x: 1 if x == 0 else 0)

    global isBinary
    sa = "sex"
    df = df.rename(columns={sa: "sa"})

    df = pd.get_dummies(
        df, columns=["age_cat", "c_charge_degree"], dtype=float, drop_first=True
    )

    dependent = "two_year_recid"

    X_train, X_test = _split_with_output(df, dependent)

    isBinary = True

    return df, "compas", X_train, X_test


def make_comm():
    # seed = 42
    df = pd.read_csv(DATA_DIR / "communities.csv")
    df = df.fillna(0)
    B = "racepctblack"
    W = "racePctWhite"
    A = "racePctAsian"
    H = "racePctHisp"
    sens_features = [2, 3, 4, 5]
    df_sens = df.iloc[:, sens_features]

    maj = majority_pop(df_sens)

    a = maj.map({B: 0, W: 1, A: 0, H: 0})

    df["race"] = a
    df = df.drop(H, axis=1)
    df = df.drop(B, axis=1)
    df = df.drop(W, axis=1)
    df = df.drop(A, axis=1)

    global isBinary
    dependent = "ViolentCrimesPerPop"
    sa = "race"

    df = df.rename(columns={sa: "sa"})

    X_train, X_test = _split_with_output(df, dependent)

    isBinary = False

    return df, "comm", X_train, X_test


def make_lsac():
    df = pd.read_csv(DATA_DIR / "lawschool.csv")
    df = df.dropna()

    df["race"] = [int(race == 7.0) for race in df["race"]]
    y = df["ugpa"]
    df["ugpa"] = np.array(y / max(y))

    df["gender"] = df["gender"].map({"male": 1, "female": 0})
    df["bar1"] = [int(grade == "P") for grade in df["bar1"]]

    global isBinary
    dependent = "ugpa"
    sa = "race"

    df = df.rename(columns={sa: "sa"})

    X_train, X_test = _split_with_output(df, dependent)

    isBinary = False

    return df, "lsac", X_train, X_test


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
WEIGHT_FORMULA = "eq15"
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
        rows_a, rows_b, labels = (
            train_vals[idx_a[valid_mask]],
            train_vals[idx_b[valid_mask]],
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
        rows_a = rows_a[valid_mask]
        rows_b = rows_b[valid_mask]
        labels = np.where(diffs[valid_mask] > 0, 1, -1)

    if is_scut:
        feats_a = np.stack(rows_a[:, pixels_idx]).astype(np.float32)
        feats_b = np.stack(rows_b[:, pixels_idx]).astype(np.float32)
        train_pairs = [
            {
                "A": fa,
                "B": fb,
                "Label": int(lbl),
                "SA_A": int(ra[sa_idx]),
                "SA_B": int(rb[sa_idx]),
            }
            for fa, fb, lbl, ra, rb in zip(feats_a, feats_b, labels, rows_a, rows_b)
        ]
    else:
        feats_a = rows_a[:, keep_cols]
        feats_b = rows_b[:, keep_cols]
        train_pairs = [
            {
                "A": fa.tolist(),
                "B": fb.tolist(),
                "Label": int(lbl),
                "SA_A": int(ra[sa_idx]),
                "SA_B": int(rb[sa_idx]),
            }
            for fa, fb, lbl, ra, rb in zip(feats_a, feats_b, labels, rows_a, rows_b)
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


def _load_dataset(dataset_name):
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
    return dataset_loaders[dataset_name]()


def run_experiments(num_runs=10, dataset="scut", use_all_pairs=False, num_comp_train=1):
    results = []
    output_df_name = None
    output_nc = 0

    for _ in range(num_runs):
        _, df_name, train, test = _load_dataset(dataset)
        output_df_name = df_name
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        y_test = test[col].values
        test_features = test.drop(columns=[col])
        is_scut = "scut" in df_name

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

        data_tr_encoder = pd.DataFrame(train_pairs)
        output_nc = len(data_tr_encoder)
        pair_meta_df = pd.DataFrame(pair_meta)
        weights = _compute_pair_weights(pair_meta_df)

        dual_encoder = Classification.train_model(
            train=data_tr_encoder,
            val=None,
            y_true=data_tr_encoder["Label"].tolist(),
            shared=True,
            epochs=100,
            df_name=df_name,
        )
        dual_encoder_weighted = Classification.train_model(
            train=data_tr_encoder,
            val=None,
            y_true=data_tr_encoder["Label"].tolist(),
            shared=True,
            epochs=100,
            train_weights=weights,
            val_weights=None,
            df_name=df_name,
        )
        dual_encoder_fair = Classification.train_model(
            train=data_tr_encoder,
            val=None,
            y_true=data_tr_encoder["Label"].tolist(),
            shared=True,
            epochs=100,
            df_name=df_name,
            fairness_lambda=FAIRNESS_LAMBDA,
        )

        y_train = train[col]
        train = train.drop(columns=[col])
        test = test.drop(columns=[col])

        if is_scut:
            accuracy_lr = f1_score_lr = AOD_lr = EOD_lr = np.nan
            mse_lr = spearman_lr = pearson_lr = I_sep_lr = np.nan
        else:
            if isBinary:
                clf = LogisticRegression().fit(train, y_train)
            else:
                clf = LinearRegression().fit(train, y_train)
            predictions_lr = clf.predict(test)
            m_lr = Metrics(y_test, predictions_lr)

            if isBinary:
                accuracy_lr = accuracy_score(y_test, predictions_lr)
                f1_score_lr = f1_score(y_test, predictions_lr)
                AOD_lr = m_lr.AOD(test["sa"])
                EOD_lr = m_lr.EOD(test["sa"])
            else:
                mse_lr = m_lr.mse()
                spearman_lr = m_lr.spearmanr_coefficient()
                pearson_lr = m_lr.pearsonr_coefficient()
            I_sep_lr = m_lr.MI_con_info(test["sa"])

        if is_scut:
            test_vals = np.stack(test_features["pixels"].values).astype(np.float32)
            predictions = batched_score(dual_encoder, test_vals, batch_size=4)
            predictions_weighted = batched_score(
                dual_encoder_weighted, test_vals, batch_size=4
            )
            predictions_fair = batched_score(dual_encoder_fair, test_vals, batch_size=4)
        else:
            test_vals = test_features.values
            predictions = dual_encoder.score(test_vals).numpy().flatten()
            predictions_weighted = (
                dual_encoder_weighted.score(test_vals).numpy().flatten()
            )
            predictions_fair = dual_encoder_fair.score(test_vals).numpy().flatten()

        if isBinary:
            pred_filt = remove_outliers(predictions)
            pred_w_filt = remove_outliers(predictions_weighted)

            km = KMeans(n_clusters=2, n_init=10, max_iter=300, random_state=42).fit(
                pred_filt.reshape(-1, 1)
            )
            predictions_kmeans = km.predict(predictions.reshape(-1, 1))
            if km.cluster_centers_[0] > km.cluster_centers_[1]:
                predictions_kmeans = 1 - predictions_kmeans

            km_w = KMeans(n_clusters=2, n_init=10, max_iter=300, random_state=42).fit(
                pred_w_filt.reshape(-1, 1)
            )
            predictions_kmeans_weighted = km_w.predict(
                predictions_weighted.reshape(-1, 1)
            )
            if km_w.cluster_centers_[0] > km_w.cluster_centers_[1]:
                predictions_kmeans_weighted = 1 - predictions_kmeans_weighted

            pred_fair_filt = remove_outliers(predictions_fair)
            km_fair = KMeans(
                n_clusters=2, n_init=10, max_iter=300, random_state=42
            ).fit(pred_fair_filt.reshape(-1, 1))
            predictions_kmeans_fair = km_fair.predict(predictions_fair.reshape(-1, 1))
            if km_fair.cluster_centers_[0] > km_fair.cluster_centers_[1]:
                predictions_kmeans_fair = 1 - predictions_kmeans_fair
        else:
            predictions_kmeans = predictions
            predictions_kmeans_weighted = predictions_weighted
            predictions_kmeans_fair = predictions_fair

        data_raw = np.column_stack([predictions_kmeans, y_test, test["sa"].values])
        data_w_raw = np.column_stack(
            [predictions_kmeans_weighted, y_test, test["sa"].values]
        )
        data_fair_raw = np.column_stack(
            [predictions_kmeans_fair, y_test, test["sa"].values]
        )

        violate_comp = violate_comp_w = violate_comp_fair = 0
        violate = violate_w = violate_fair = 0
        n_rows = len(data_raw)
        y_vals = data_raw[:, 1]
        has_comparable_pairs = np.unique(y_vals).size > 1
        half_size = n_rows // 2

        for _ in range(r):
            selectedr = np.random.choice(n_rows, size=half_size, replace=False)
            ps = separation(
                data_raw[selectedr, 1], data_raw[selectedr, 0], data_raw[selectedr, 2]
            )
            ps_w = separation(
                data_w_raw[selectedr, 1],
                data_w_raw[selectedr, 0],
                data_w_raw[selectedr, 2],
            )
            ps_fair = separation(
                data_fair_raw[selectedr, 1],
                data_fair_raw[selectedr, 0],
                data_fair_raw[selectedr, 2],
            )
            violate += min(ps) < alpha
            violate_w += min(ps_w) < alpha
            violate_fair += min(ps_fair) < alpha

        for _ in range(r):
            if not has_comparable_pairs:
                continue

            i1 = np.empty(n_rows, dtype=np.int64)
            i2 = np.empty(n_rows, dtype=np.int64)
            filled = 0
            while filled < n_rows:
                remaining = n_rows - filled
                batch_size = max(remaining * 2, 256)
                idx1 = np.random.randint(0, n_rows, size=batch_size)
                idx2 = np.random.randint(0, n_rows, size=batch_size)
                valid = y_vals[idx1] != y_vals[idx2]
                if not np.any(valid):
                    continue
                take = min(remaining, int(valid.sum()))
                i1[filled : filled + take] = idx1[valid][:take]
                i2[filled : filled + take] = idx2[valid][:take]
                filled += take

            violate_comp += (
                min(comparative_separation(count_violation_fast(data_raw, i1, i2))[0])
                < alpha
            )
            violate_comp_w += (
                min(comparative_separation(count_violation_fast(data_w_raw, i1, i2))[0])
                < alpha
            )
            violate_comp_fair += (
                min(
                    comparative_separation(count_violation_fast(data_fair_raw, i1, i2))[
                        0
                    ]
                )
                < alpha
            )

        m_bi = Metrics(y_test, predictions_kmeans)
        m_weighted_bi = Metrics(y_test, predictions_kmeans_weighted)
        m_fair_bi = Metrics(y_test, predictions_kmeans_fair)
        I_sep_bi = m_bi.MI_con_info(test["sa"])
        I_sep_weighted_bi = m_weighted_bi.MI_con_info(test["sa"])
        I_sep_fair_bi = m_fair_bi.MI_con_info(test["sa"])

        if isBinary:
            result = {
                "weight_formula": WEIGHT_FORMULA,
                "fairness_lambda": FAIRNESS_LAMBDA,
                "Acc_lr": accuracy_lr,
                "Acc_unweight": accuracy_score(y_test, predictions_kmeans),
                "Acc_weighted": accuracy_score(y_test, predictions_kmeans_weighted),
                "Acc_fairreg": accuracy_score(y_test, predictions_kmeans_fair),
                "F1_lr": f1_score_lr,
                "F1_unweight": f1_score(y_test, predictions_kmeans),
                "F1_weighted": f1_score(y_test, predictions_kmeans_weighted),
                "F1_fairreg": f1_score(y_test, predictions_kmeans_fair),
                "AOD_lr": AOD_lr,
                "AOD_unweight": m_bi.AOD(test["sa"]),
                "AOD_weighted": m_weighted_bi.AOD(test["sa"]),
                "AOD_fairreg": m_fair_bi.AOD(test["sa"]),
                "EOD_lr": EOD_lr,
                "EOD_unweight": m_bi.EOD(test["sa"]),
                "EOD_weighted": m_weighted_bi.EOD(test["sa"]),
                "EOD_fairreg": m_fair_bi.EOD(test["sa"]),
                "I_sep_lr": I_sep_lr,
                "I_sep_bi": I_sep_bi,
                "I_sep_weighted_bi": I_sep_weighted_bi,
                "I_sep_fairreg_bi": I_sep_fair_bi,
                "violate_r": violate / r,
                "violate_r_weighted": violate_w / r,
                "violate_r_fairreg": violate_fair / r,
                "violate_comp_r": violate_comp / r,
                "violate_comp_r_w": violate_comp_w / r,
                "violate_comp_r_fairreg": violate_comp_fair / r,
            }
        else:
            result = {
                "weight_formula": WEIGHT_FORMULA,
                "fairness_lambda": FAIRNESS_LAMBDA,
                "MSE_lr": mse_lr,
                "MSE_unweight": m_bi.mse(),
                "MSE_weight": m_weighted_bi.mse(),
                "MSE_fairreg": m_fair_bi.mse(),
                "spearman_lr": spearman_lr,
                "spearman_unweighted": m_bi.spearmanr_coefficient(),
                "spearman_weighted": m_weighted_bi.spearmanr_coefficient(),
                "spearman_fairreg": m_fair_bi.spearmanr_coefficient(),
                "pearson_lr": pearson_lr,
                "pearson_unweighted": m_bi.pearsonr_coefficient(),
                "pearson_weighted": m_weighted_bi.pearsonr_coefficient(),
                "pearson_fairreg": m_fair_bi.pearsonr_coefficient(),
                "I_sep_lr": I_sep_lr,
                "I_sep_bi": I_sep_bi,
                "I_sep_weighted_bi": I_sep_weighted_bi,
                "I_sep_fairreg_bi": I_sep_fair_bi,
                "violate_r": violate / r,
                "violate_r_weighted": violate_w / r,
                "violate_r_fairreg": violate_fair / r,
                "violate_comp_r": violate_comp / r,
                "violate_comp_r_w": violate_comp_w / r,
                "violate_comp_r_fairreg": violate_comp_fair / r,
            }
        results.append(result)

    pair_strategy = "all" if use_all_pairs else str(num_comp_train)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(
        RESULTS_DIR
        / f"FairReweighing_violate_r_{output_df_name}_{output_nc}_{pair_strategy}.csv",
        index=False,
    )


if __name__ == "__main__":
    DATASET = "comm"  # scut, adult, german, heart, compas, comm, lsac
    NUM_RUNS = 10
    USE_ALL_PAIRS = False
    NUM_COMP_TRAIN = 20
    run_experiments(
        num_runs=NUM_RUNS,
        dataset=DATASET,
        use_all_pairs=USE_ALL_PAIRS,
        num_comp_train=NUM_COMP_TRAIN,
    )
