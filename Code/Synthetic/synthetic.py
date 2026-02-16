from collections import Counter
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

    X = df[["pixels", "sa"]]

    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    X_train[col] = y_train
    X_test[col] = y_test

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

    X = df.drop([dependent, "education", "native-country"], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    X_train[col] = y_train
    X_test[col] = y_test

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

    X = df.drop([dependent], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    X_train[col] = y_train
    X_test[col] = y_test

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

    X = df.drop([dependent], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    X_train[col] = y_train
    X_test[col] = y_test

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
    # sensitive attribute names
    A = ["sex", "race"]
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

    X = df.drop([dependent], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    X_train[col] = y_train
    X_test[col] = y_test

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

    X = df.drop([dependent], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    X_train[col] = y_train
    X_test[col] = y_test

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

    X = df.drop([dependent], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    X_train[col] = y_train
    X_test[col] = y_test

    isBinary = False

    return df, "lsac", X_train, X_test


def majority_pop(a):
    B = "racepctblack"
    W = "racePctWhite"
    A = "racePctAsian"
    H = "racePctHisp"
    maj = a.apply(pd.Series.idxmax, axis=1)
    return maj


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
    outliers = X[(X < lower_bound) | (X > upper_bound)]

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
    count = []
    for i in range(len(s)):
        count.append(str(int(y_pred[i])) + str(int(y[i])) + str(int(s[i])))
    x = Counter(count)

    mut1, vart1 = stats(x["111"], x["011"])
    mut0, vart0 = stats(x["110"], x["010"])

    eps = np.finfo(float).eps
    zt = (mut1 - mut0) / np.sqrt(max(vart1 + vart0, eps))
    pt = norm.sf(np.abs(zt)) * 2

    muf1, varf1 = stats(x["101"], x["001"])
    muf0, varf0 = stats(x["100"], x["000"])
    zf = (muf1 - muf0) / np.sqrt(max(varf1 + varf0, eps))
    pf = norm.sf(np.abs(zf)) * 2

    return [pt, pf]


results = []
use_all_pairs = True  # Set to True to use all possible pairs (N^2)

alpha = 0.05
r = 100
nc = 1000
num_comp_train = 1
num_comp_test = 1
WEIGHT_FORMULA = "eq15"

for i in range(10):
    df, df_name, train, test = make_german()
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    y_test = test["output"].values
    test_features = test.drop(columns=["output"])
    is_scut = "scut" in df_name

    # Pre-extract values for index-based access (much faster than iterrows)
    train_vals = train.values
    train_cols = list(train.columns)
    col_idx, sa_idx = train_cols.index(col), train_cols.index("sa")
    pixels_idx = train_cols.index("pixels") if is_scut else None

    res_tr_encoder = []
    svc_encoder = []
    res_tr_sa = []

    res_ts_encoder = []
    res_ts_sa = []

    if use_all_pairs:
        # Generate all directed pairs (i, j) where i != j.
        num_train = len(train_vals)
        idx_a, idx_b = np.where(~np.eye(num_train, dtype=bool))

        # Vectorized labels for all pairs
        labels = np.sign(
            train_vals[idx_a, col_idx] - train_vals[idx_b, col_idx]
        ).astype(int)

        # Filter out ties (Label 0)
        valid_mask = labels != 0
        idx_a, idx_b, labels = idx_a[valid_mask], idx_b[valid_mask], labels[valid_mask]

        # Vectorized feature extraction
        rows_a = train_vals[idx_a]
        rows_b = train_vals[idx_b]

        # Create mask for columns to keep (all except col_idx)
        keep_cols = np.ones(train_vals.shape[1], dtype=bool)
        keep_cols[col_idx] = False

        if is_scut:
            feats_a = np.stack(rows_a[:, pixels_idx]).astype(np.float32)
            feats_b = np.stack(rows_b[:, pixels_idx]).astype(np.float32)
        else:
            feats_a = rows_a[:, keep_cols]
            feats_b = rows_b[:, keep_cols]

        # Batch create dictionaries
        if is_scut:
            res_tr_encoder = [
                {"A": fa, "B": fb, "Label": int(lbl)}
                for fa, fb, lbl in zip(feats_a, feats_b, labels)
            ]
        else:
            res_tr_encoder = [
                {"A": fa.tolist(), "B": fb.tolist(), "Label": int(lbl)}
                for fa, fb, lbl in zip(feats_a, feats_b, labels)
            ]

        res_tr_sa = [
            {"AB": (ra[sa_idx], rb[sa_idx]), "AY": ((ra[sa_idx], rb[sa_idx]), int(lbl))}
            for ra, rb, lbl in zip(rows_a, rows_b, labels)
        ]
    else:
        # Random pair generation with both directions for each selected pair.
        n_train = len(train)
        keep_cols = np.ones(train_vals.shape[1], dtype=bool)
        keep_cols[col_idx] = False

        # Pre-allocate arrays for batch processing
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
                used_pairs.add((min(idx_a, int(idx_b)), max(idx_a, int(idx_b))))
                # Add both directed pairs once: (i, j) and (j, i).
                all_idx_a.append(idx_a)
                all_idx_b.append(int(idx_b))
                all_idx_a.append(int(idx_b))
                all_idx_b.append(idx_a)

        # Batch process all pairs
        all_idx_a = np.array(all_idx_a)
        all_idx_b = np.array(all_idx_b)

        rows_a = train_vals[all_idx_a]
        rows_b = train_vals[all_idx_b]

        diffs = rows_a[:, col_idx] - rows_b[:, col_idx]
        valid_mask = diffs != 0
        labels = np.where(diffs > 0, 1, -1)

        # Filter valid pairs
        rows_a = rows_a[valid_mask]
        rows_b = rows_b[valid_mask]
        labels = labels[valid_mask]

        if is_scut:
            feats_a = np.stack(rows_a[:, pixels_idx]).astype(np.float32)
            feats_b = np.stack(rows_b[:, pixels_idx]).astype(np.float32)
        else:
            feats_a = rows_a[:, keep_cols]
            feats_b = rows_b[:, keep_cols]

        if is_scut:
            res_tr_encoder = [
                {"A": fa, "B": fb, "Label": int(lbl)}
                for fa, fb, lbl in zip(feats_a, feats_b, labels)
            ]
        else:
            res_tr_encoder = [
                {"A": fa.tolist(), "B": fb.tolist(), "Label": int(lbl)}
                for fa, fb, lbl in zip(feats_a, feats_b, labels)
            ]

        res_tr_sa = [
            {"AB": (ra[sa_idx], rb[sa_idx]), "AY": ((ra[sa_idx], rb[sa_idx]), int(lbl))}
            for ra, rb, lbl in zip(rows_a, rows_b, labels)
        ]

    # for indexA, rowA in test.iterrows():
    #     comp = []
    #     comp_count = 0
    #     test_cp = test.copy()
    #     while comp_count < num_comp_test:
    #     # for indexB, rowB in test.iterrows():
    #         rowB = test_cp.sample()
    #         indexB = rowB.index[0]
    #         if (indexB == indexA):
    #             continue
    #         rowB = rowB.iloc[0]
    #         ratingA = rowA[col]
    #         ratingB = rowB[col]
    #         label = -1
    #         if ratingA > ratingB:
    #             label = 1
    #         elif ratingA < ratingB:
    #             label = 0
    #         if label != -1:
    #             # if label is not None:
    #             testA = rowA.drop(labels=[col])
    #             testB = rowB.drop(labels=[col])
    #
    #             res_ts_encoder.append({"A": testA.to_list(),
    #                                    "B": testB.to_list(),
    #                                    })
    #
    #
    #             res_ts_sa.append({"A": testA['sa'],
    #                               "B": testB['sa'],
    #                               "Label": label})
    #
    #             test_cp.drop(indexB, inplace=True)
    #             comp_count += 1

    data_tr_encoder = pd.DataFrame(res_tr_encoder)
    nc = len(data_tr_encoder)

    # Optimized weights calculation (vectorized), Eq. (15) only.
    res_tr_sa = pd.DataFrame(res_tr_sa)
    ab_array = np.array(res_tr_sa["AB"].tolist())
    is_same_group = ab_array[:, 0] == ab_array[:, 1]

    p_aij = res_tr_sa["AB"].value_counts(normalize=True)
    p_aij_yij = res_tr_sa["AY"].value_counts(normalize=True)

    # Eq. (15): same-group pairs keep weight 1; cross-group pairs get P(aij)/(2P(aij,yij)).
    weights = np.ones(len(res_tr_sa))
    diff_group_mask = ~is_same_group
    weights[diff_group_mask] = (
        res_tr_sa.loc[diff_group_mask, "AB"].map(p_aij)
        / (2 * res_tr_sa.loc[diff_group_mask, "AY"].map(p_aij_yij))
    ).values

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

    # predictions = comp_pred(data_ts_encoder, dual_encoder)
    # predictions_weighted = comp_pred(data_ts_encoder, dual_encoder_weighted)

    # accuracy = accuracy_score(res_ts_sa['Label'], predictions)

    y_train = train["output"]
    train = train.drop(columns=["output"])
    test = test.drop(columns=["output"])

    # #
    # y_svc = svc_encoder['label']
    # svc_train = svc_encoder.drop(columns=['label'])
    #
    # svc = LinearSVC(fit_intercept=False, loss='hinge', max_iter=100000)
    # svc.fit(svc_train, y_svc)
    # svc_predictions = svc.predict(test)
    # svc_predictions_reg = svc.decision_function(test)
    # svc_predictions = [0 if x == -1 else 1 for x in svc_predictions]

    # accuracy_svc = accuracy_score(y_test, svc_predictions)
    # f1_score_svc = f1_score(y_test, svc_predictions)
    # m_svc = Metrics(y_test, svc_predictions_reg)
    # AOD_svc = m_svc.AOD(test['sa'])
    # EOD_svc = m_svc.EOD(test['sa'])
    # spearman_svc = m_svc.spearmanr_coefficient()
    # pearson_svc = m_svc.pearsonr_coefficient()
    # MSE_svc = m_svc.mse()
    # I_sep_svc = m_svc.MI_con_info(test['sa'])
    #
    # Skip baseline linear-model training/comparison for SCUT runs.
    if is_scut:
        accuracy_lr = np.nan
        f1_score_lr = np.nan
        AOD_lr = np.nan
        EOD_lr = np.nan
        mse_lr = np.nan
        spearman_lr = np.nan
        pearson_lr = np.nan
        I_sep_lr = np.nan
    else:
        if isBinary:
            clf = LogisticRegression().fit(train, y_train)
            predictions = clf.predict(test)
        else:
            clf = LinearRegression().fit(train, y_train)
            predictions = clf.predict(test)

        m_lr = Metrics(y_test, predictions)

        if isBinary:
            # y_score = clf.predict_proba(test)[:, 1]
            accuracy_lr = accuracy_score(y_test, predictions)
            f1_score_lr = f1_score(y_test, predictions)
            AOD_lr = m_lr.AOD(test["sa"])
            EOD_lr = m_lr.EOD(test["sa"])
        # #
        # fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_score)
        # roc_auc_lr = auc(fpr_lr, tpr_lr)
        #
        else:
            mse_lr = m_lr.mse()
            spearman_lr = m_lr.spearmanr_coefficient()
            pearson_lr = m_lr.pearsonr_coefficient()

        I_sep_lr = m_lr.MI_con_info(test["sa"])

    # Batch Prediction (Single call for better performance)
    if is_scut:
        test_vals = np.stack(test_features["pixels"].values).astype(np.float32)
    else:
        test_vals = test_features.values
    if is_scut:
        predictions = batched_score(dual_encoder, test_vals, batch_size=16)
        predictions_weighted = batched_score(
            dual_encoder_weighted, test_vals, batch_size=16
        )
    else:
        predictions = dual_encoder.score(test_vals).numpy().flatten()
        predictions_weighted = dual_encoder_weighted.score(test_vals).numpy().flatten()

    if not isBinary:
        predictions_kmeans = predictions
        predictions_kmeans_weighted = predictions_weighted

    else:
        # Outlier removal and KMeans (Optimized)
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
        predictions_kmeans_weighted = km_w.predict(predictions_weighted.reshape(-1, 1))
        if km_w.cluster_centers_[0] > km_w.cluster_centers_[1]:
            predictions_kmeans_weighted = 1 - predictions_kmeans_weighted

    # Optimized Simulation Loop (The biggest bottleneck)
    data_raw = np.column_stack([predictions_kmeans, y_test, test["sa"].values])
    data_w_raw = np.column_stack(
        [predictions_kmeans_weighted, y_test, test["sa"].values]
    )

    violate_comp = violate_comp_w = 0
    violate = violate_w = 0
    n_rows = len(data_raw)
    y_vals = data_raw[:, 1]
    has_comparable_pairs = np.unique(y_vals).size > 1

    # Vectorize violation checks
    half_size = n_rows // 2
    for _ in range(r):
        selectedr = np.random.choice(n_rows, size=half_size, replace=False)
        ps = separation(
            data_raw[selectedr, 1], data_raw[selectedr, 0], data_raw[selectedr, 2]
        )
        ps_w = separation(
            data_w_raw[selectedr, 1], data_w_raw[selectedr, 0], data_w_raw[selectedr, 2]
        )
        violate += min(ps) < alpha
        violate_w += min(ps_w) < alpha

    for _ in range(r):
        if not has_comparable_pairs:
            continue

        # Build exactly n_rows pairs with Y1 != Y2.
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
            i1[filled: filled + take] = idx1[valid][:take]
            i2[filled: filled + take] = idx2[valid][:take]
            filled += take

        violate_comp += (
            min(comparative_separation(count_violation_fast(data_raw, i1, i2))[0])
            < alpha
        )
        violate_comp_w += (
            min(comparative_separation(count_violation_fast(data_w_raw, i1, i2))[0])
            < alpha
        )
    # combined_unweighted = np.column_stack((predictions, predictions_kmeans))
    # combined_weighted = np.column_stack((predictions_weighted, predictions_kmeans_weighted))
    #
    # df_unweighted = pd.DataFrame(combined_unweighted, columns=['predictions', 'kmeans'])
    # df_weighted = pd.DataFrame(combined_weighted, columns=['predictions', 'kmeans'])
    #
    # group0_unweighted = df_unweighted[df_unweighted['kmeans'] == 0]['predictions']
    # group1_unweighted = df_unweighted[df_unweighted['kmeans'] == 1]['predictions']
    #
    # group0_weighted = df_weighted[df_weighted['kmeans'] == 0]['predictions']
    # group1_weighted = df_weighted[df_weighted['kmeans'] == 1]['predictions']

    # fpr, tpr, thresholds = roc_curve(y_test, predictions)
    # roc_auc = auc(fpr, tpr)
    #
    # fpr_weighted, tpr_weighted, threshold_weighted = roc_curve(y_test, predictions_weighted)
    # roc_auc_weighted = auc(fpr_weighted, tpr_weighted)

    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'ROC_lr curve (area = {roc_auc_lr:.2f})')
    # plt.plot(fpr_weighted, tpr_weighted, color='red', lw=2, label=f'ROC_weighted curve (area = {roc_auc_weighted:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show()

    # res_index = next(x for x, val in enumerate(tpr) if val >= 0.8)
    # res_index_weighted = next(x for x, val in enumerate(tpr_weighted) if val >= 0.8)
    #
    # threshold = thresholds[res_index]
    # threshold_weighted = threshold_weighted[res_index_weighted]

    # plt.hist(group0_unweighted, bins=np.linspace(-1, 1, 50), label='Group 0')
    # plt.hist(group1_unweighted, bins=np.linspace(-1, 1, 50), label='Group 1')
    # plt.scatter(kmeans_unweighted.cluster_centers_,[0,0], c='r')
    # plt.scatter(threshold,0,c='g', label='ROC Threshold')
    # plt.legend()
    # plt.title('predictions on test data')
    # plt.show()
    #
    # plt.hist(group0_weighted, bins=np.linspace(-1, 1, 50), label='Group 0')
    # plt.hist(group1_weighted, bins=np.linspace(-1, 1, 50), label='Group 1')
    # plt.scatter(kmeans_weighted.cluster_centers_,[0,0], c='r')
    # plt.scatter(threshold_weighted,0,c='g', label='ROC Threshold')
    # plt.legend()
    # plt.title('predictions on weighted test data')
    # plt.show()

    #
    # predictions_bi = []
    # predictions_weighted_bi = []
    #
    # for index, item in enumerate(predictions):
    #     if item >= threshold:
    #         predictions_bi.append(1)
    #     else:
    #         predictions_bi.append(0)
    #
    # for index, item in enumerate(predictions_weighted):
    #     if item >= threshold_weighted:
    #         predictions_weighted_bi.append(1)
    #     else:
    #         predictions_weighted_bi.append(0)

    # m = Metrics(y_test, predictions)
    m_bi = Metrics(y_test, predictions_kmeans)
    m_weighted_bi = Metrics(y_test, predictions_kmeans_weighted)

    if isBinary:
        accuracy_bi = accuracy_score(y_test, predictions_kmeans)
        accuracy_weighted = accuracy_score(y_test, predictions_kmeans_weighted)
        f1_score_bi = f1_score(y_test, predictions_kmeans)
        f1_score_weighted = f1_score(y_test, predictions_kmeans_weighted)

        AOD = m_bi.AOD(test["sa"])
        EOD = m_bi.EOD(test["sa"])
        AOD_weighted = m_weighted_bi.AOD(test["sa"])
        EOD_weighted = m_weighted_bi.EOD(test["sa"])

    else:
        MSE_unweighted = m_bi.mse()
        MSE_weighted = m_weighted_bi.mse()

        spearman_unweighted = m_bi.spearmanr_coefficient()
        pearson_unweighted = m_bi.pearsonr_coefficient()

        spearman_weighted = m_weighted_bi.spearmanr_coefficient()
        pearson_weighted = m_weighted_bi.pearsonr_coefficient()

    I_sep_bi = m_bi.MI_con_info(test["sa"])
    I_sep_weighted_bi = m_weighted_bi.MI_con_info(test["sa"])

    if isBinary:
        result = {
            "weight_formula": WEIGHT_FORMULA,
            "Acc_lr": accuracy_lr,
            "Acc_unweight": accuracy_bi,
            "Acc_weighted": accuracy_weighted,
            "F1_lr": f1_score_lr,
            "F1_unweight": f1_score_bi,
            "F1_weighted": f1_score_weighted,
            "AOD_lr": AOD_lr,
            "AOD_unweight": AOD,
            "AOD_weighted": AOD_weighted,
            "EOD_lr": EOD_lr,
            "EOD_unweight": EOD,
            "EOD_weighted": EOD_weighted,
            "I_sep_lr": I_sep_lr,
            "I_sep_bi": I_sep_bi,
            "I_sep_weighted_bi": I_sep_weighted_bi,
            "violate_r": violate / r,
            "violate_r_weighted": violate_w / r,
            "violate_comp_r": violate_comp / r,
            "violate_comp_r_w": violate_comp_w / r,
        }
    else:
        result = {
            "weight_formula": WEIGHT_FORMULA,
            "MSE_lr": mse_lr,
            "MSE_unweight": MSE_unweighted,
            "MSE_weight": MSE_weighted,
            "spearman_lr": spearman_lr,
            "spearman_unweighted": spearman_unweighted,
            "spearman_weighted": spearman_weighted,
            "pearson_lr": pearson_lr,
            "pearson_unweighted": pearson_unweighted,
            "pearson_weighted": pearson_weighted,
            "I_sep_lr": I_sep_lr,
            "I_sep_bi": I_sep_bi,
            "I_sep_weighted_bi": I_sep_weighted_bi,
            "violate_r": violate / r,
            "violate_r_weighted": violate_w / r,
            "violate_comp_r": violate_comp / r,
            "violate_comp_r_w": violate_comp_w / r,
        }

    results.append(result)

pair_strategy = "all" if use_all_pairs else str(num_comp_train)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(
    RESULTS_DIR / f"FairReweighing_violate_r_{df_name}_{nc}_{pair_strategy}.csv",
    index=False,
)

# changed encoder structure
# use one pair for every training entry
# Compare AUC, AOD, EOD with logistic regression
# Build models on the whole adult dataset
# I_sep when comparing linea output
# Switch to German and Heart
# including accuracy (F1,precision...) and mAOD from FairBalance
# try linearSVC/ SVC (fit_intercept=False, loss='hinge') with entry A minus entry B, make prediction on individual and calculate accuracy.
# bin continuous results and plot a histogram on the training data
# focus more on regression datasets like communities, lsac and SCUT
# put FairReiweghing paper on arxiv

# fix cluster labeling, plot historgram with cutoff
# finding different ways for clustering (KDE, SVM)
# historgram for training and test data and both
# cutoff with ROC curve
# sample weight implementation

# increasing the numbers of comparison for better accuracy (german & heart)
# check MSE, spearman for regression
# new ways to preprocessing reweighing

# work on comp_sep paper & github repo
# real world dataset
# test comp_sep on comp FairReweighing

# select the same testing pairs for weighted and unweighted
# trying using all pairs
# report average difference between TPR and FPR
# testing with difference number of nc
# focus on the paper first (introduction, background)


# split the dataset in half stratifily on SA
# train a classification model and one with fairbalance
# sample without replacement for a certain number of repetition

# include COMPAS and regression dataset
# include SCUT and other fairness metrics

# include the hypothesis test for seperation
# getting rid of the validation and earlystopping


# for i in range(10):
# # m = Metrics(df["income"], df["pred"])
# # AOD = m.AOD(df["gender"])
# # EOD = m.EOD(df["gender"])
# # gAOD = m.gAOD(df["gender"])
# # MI = m.MI_b(df["gender"])
#
#     res_ts_encoder = []
# test_list = []
#
# for indexA, rowA in test.iterrows():
#     comp = []
# test_cp = test.copy()
# comp_count = 0
# while comp_count < num_comp_test:
#     rowB = test_cp.sample()
#     indexB = rowB.index[0]
#     if (indexB == indexA):
#         continue
#     rowB = rowB.iloc[0]
#     ratingA = rowA[col]
#     ratingB = rowB[col]
#     label = 0
#     if ratingA > ratingB:
#         label = 1
#     elif ratingA < ratingB:
#         label = -1
#     if label != 0:
#         # if label is not None:
#         testA = rowA.drop(labels=[col])
#         testB = rowB.drop(labels=[col])
#
#         res_ts_encoder.append({"A": testA.to_list(),
#                                "B": testB.to_list(),
#                                "Label": label
#                                })
#         test_list.append({"A": testA['sa'],
#                           "B": testB['sa'],
#                           "Label": label
#                           })
#         test_cp.drop(indexB, inplace=True)
#         comp_count += 1
#
# data_ts_encoder = pd.DataFrame(res_ts_encoder)
# test_list = pd.DataFrame(test_list)
#
# predictions = Classification.predict(train_encoder, dual_encoder)

# res_tr = []
# comp = []
# for indexA in range(0, len(df)):
#     df_cp = df.copy()
#     comp_count = 0
#     rowA = df.iloc[indexA]
#     while comp_count < num_comp:
#         # for indexB in range(0, len(df)):
#         #     indexB = random.randint(0, len(df_cp) - 1)
#         rowB = df_cp.sample()
#         indexB = rowB.index[0]
#         if (indexB == indexA):
#                 continue
#         rowB = rowB.iloc[0]
#         ratingA = rowA[col]
#         ratingB = rowB[col]
#         predA = rowA["pred"]
#         predB = rowB["pred"]
#         label = 0
#         pred = 0
#         if ratingA > ratingB:
#             label = 1
#         elif ratingA < ratingB:
#             label = -1
#         if predA > predB:
#             pred = 1
#         elif predA < predB:
#             pred = -1
#         res_tr.append({"A": rowA["gender"],
#                        "B": rowB["gender"],
#                        "Label": label,
#                        "pred_con": pred
#                        })
#         # comp.append(indexB)
#         df_cp.drop(indexB, inplace=True)
#         comp_count += 1

# data_tr = pd.DataFrame(res_tr)

# test_list["pred"] = predictions
# m = Metrics(test_list["Label"], test_list["pred"])
# AOD_comp = m.AOD_comp(test_list[["A", "B"]])
# Within_comp = m.Within_comp(test_list[["A", "B"]])
# Sep_comp = m.Sep_comp(test_list[["A", "B"]])
# # MI_comp = m.MI_comp(data_tr[["A", "B"]])
# # MI_comp2 = m.MI_comp2(data_tr[["A", "B"]])
#
# result = {"# of comparisons": len(test_list), "AOD_comp": AOD_comp,
#           "Within_comp": Within_comp, "EOD_comp": AOD_comp + Within_comp,
#           # "MI_comp": MI_comp, "MI_comp2": MI_comp2, "Ratio": MI / MI_comp
#           }
# results.append(result)
#
# results = pd.DataFrame(results)
# results.loc[len(results.index)] = results.mean()
# results.loc[len(results.index)] = results.std()
# results.to_csv(df_name + "_encoder_" + str(num_comp_train) + '_' + str(num_comp_test) + ".csv", index=False)

# experiment with the num of comparison (repeat 20 times and get mean and std)
# repeated trail on df1-df3 and add more data points
# SCUT dataset
# Find out the relationship between num of comparison and num of data points
