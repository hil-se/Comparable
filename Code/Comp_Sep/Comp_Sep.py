from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import DistanceMetric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from metrics import Metrics

# split the dataset in half stratifily on SA
# train a classification model and one with fairbalance
# sample without replacement for a certain number of repetition

# try different SA on adult and heart
# try maintaining 2xnum_test
# experiment on community and crime dataset
# benchmmark on the training set

# include hypothesis testing on AOD and EOD
# start with n = num_test
# sampling testing data and calculate seperation violation and comp_sep
# experiment on community and crime dataset

col = "output"

alpha = 0.05
r = 100
nc = 2000


def make_german():
    df = pd.read_csv("../../Data/german_credit_data.csv", index_col=0)
    df = df.dropna()
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == "male" else 0)
    df['Risk'] = df['Risk'].apply(lambda x: 1 if x == "good" else 0)

    dependent = 'Risk'
    sa = 'Sex'

    df = df.rename(columns={sa: 'sa'})

    df = pd.get_dummies(df, columns=['Housing', 'Saving accounts', 'Checking account', 'Purpose'], dtype=float,
                        drop_first=True)

    X = df.drop([dependent], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5)

    X_train[col] = y_train
    X_test[col] = y_test

    return df, "german", X_train, X_test, sa


def make_heart(sa='sex'):
    df = pd.read_csv("../../Data/heart.csv")
    df = df.dropna()
    if sa == 'age':
        df['age'] = df['age'].apply(lambda x: 1 if x >= 55 else 0)

    dependent = 'output'

    df = df.rename(columns={sa: 'sa'})

    X = df.drop([dependent], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5)

    X_train[col] = y_train
    X_test[col] = y_test

    return df, "heart", X_train, X_test, sa


def make_adult(sa='gender'):
    df = pd.read_csv("../../Data/adult.csv", na_values=["?"])
    df = df.dropna()
    df['gender'] = df['gender'].apply(lambda x: 1 if x == "Male" else 0)
    df['income'] = df['income'].apply(lambda x: 1 if x == ">50K" else 0)
    df['race'] = df['race'].apply(lambda x: 1 if x == "White" else 0)

    dependent = 'income'

    df = df.rename(columns={sa: 'sa'})

    df = pd.get_dummies(df, columns=['workclass', 'marital-status', 'occupation',
                                     'relationship'], dtype=float,
                        drop_first=True)

    X = df.drop([dependent, 'education', 'native-country'], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=df['sa'])

    X_train[col] = y_train
    X_test[col] = y_test

    return df, "adult", X_train, X_test, sa


def make_comm():
    df = pd.read_csv("../../Data/communities.csv")
    df = df.fillna(0)
    B = "racepctblack"
    W = "racePctWhite"
    A = "racePctAsian"
    H = "racePctHisp"
    sens_features = [2, 3, 4, 5]
    df_sens = df.iloc[:, sens_features]

    maj = majority_pop(df_sens)

    a = maj.map({B: 0, W: 1, A: 0, H: 0})

    df['race'] = a
    df = df.drop(H, axis=1)
    df = df.drop(B, axis=1)
    df = df.drop(W, axis=1)
    df = df.drop(A, axis=1)

    dependent = 'ViolentCrimesPerPop'
    sa = 'race'

    df = df.rename(columns={sa: 'sa'})

    X = df.drop([dependent], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5)

    X_train[col] = y_train
    X_test[col] = y_test

    return df, "comm", X_train, X_test, sa

def make_lsac():
    df = pd.read_csv("../../Data/lawschool.csv")
    df = df.dropna()

    df['race'] = [int(race == 7.0) for race in df['race']]
    y = df['ugpa']
    df['ugpa'] = np.array(y / max(y))

    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    df['bar1'] = [int(grade == 'P') for grade in df['bar1']]

    dependent = 'ugpa'
    sa = 'race'

    df = df.rename(columns={sa: 'sa'})

    X = df.drop([dependent], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5)

    X_train[col] = y_train
    X_test[col] = y_test

    return df, "lsac", X_train, X_test, sa



def majority_pop(a):
    B = "racepctblack"
    W = "racePctWhite"
    A = "racePctAsian"
    H = "racePctHisp"
    maj = a.apply(pd.Series.idxmax, axis=1)
    return maj


def density(X, isContinuous=False):
    """Calculates density efficiently using pandas value_counts."""
    if isinstance(X, list):
        X = np.array(X)

    if isContinuous:
        scaler = StandardScaler()
        if X.ndim == 1:
            X_z = scaler.fit_transform(X.reshape(-1, 1))
        else:
            X_z = scaler.fit_transform(X)
        dists = DistanceMetric.get_metric('euclidean').pairwise(X_z)
        return np.sum(dists < 0.5, axis=1)

    elif X.ndim == 1:
        s = pd.Series(X)
        probs = s.value_counts(normalize=True)
        return s.map(probs).to_numpy()
    else:
        df = pd.DataFrame(X)
        # Calculate probability for each unique row
        counts = df.value_counts(normalize=True, sort=False)
        # Map probabilities back to the original rows
        # merging on columns preserves the mapping
        df_merged = df.merge(counts.rename('prob'), left_on=list(df.columns), right_index=True, how='left')
        return df_merged['prob'].to_numpy()


def weight(A, y, treatment="FairBalance", isContinuous=False):
    X = np.stack((A, y), axis=1)

    w = density(X, isContinuous)
    wA = density(A, isContinuous)
    wy = density(y, isContinuous)

    if treatment == "FairBalanceVariant":
        weights = 1 / w
    elif treatment == "FairBalance":
        weights = wA / w
    elif treatment == "GroupBalance":
        weights = wy / w
    elif treatment == "Reweighing":
        weights = wA * wy / w
    else:
        return None

    weights = (len(weights) * weights / sum(weights)).flatten()
    return weights


def cal_comp(xt1, x1, xt2, x2):
    denom = x1 + x2
    if denom == 0:
        return 0.0, 0.0
    mu = (xt1 + xt2) / denom
    var = mu * (1 - mu) / denom
    return mu, var

def stats(x1, x2):
        mu = x1 / (x1+x2)
        var = mu*(1-mu) / (x1+x2)
        return mu, var

def separation(x):
    mut1, vart1 = stats(x["111"], x["011"])
    mut0, vart0 = stats(x["110"], x["010"])
    zt = (mut1 - mut0) / np.sqrt(vart1 + vart0)
    pt = norm.sf(np.abs(zt)) * 2
    muf1, varf1 = stats(x["101"], x["001"])
    muf0, varf0 = stats(x["100"], x["000"])
    zf = (muf1 - muf0) / np.sqrt(varf1 + varf0)
    pf = norm.sf(np.abs(zf)) * 2
    return [pt, pf]


def comparative_separation(x):
    mut11, vart11 = cal_comp(x["1111"], x["1111"] + x["0111"] + x["x111"], x["0011"], x["0011"] + x["1011"] + x["x011"])
    mut00, vart00 = cal_comp(x["1100"], x["1100"] + x["0100"] + x["x100"], x["0000"], x["0000"] + x["1000"] + x["x000"])
    mut10, vart10 = cal_comp(x["1110"], x["1110"] + x["0110"] + x["x110"], x["0001"], x["0001"] + x["1001"] + x["x001"])
    mut01, vart01 = cal_comp(x["1101"], x["1101"] + x["0101"] + x["x101"], x["0010"], x["0010"] + x["1010"] + x["x010"])

    denom_c = np.sqrt(vart10 + vart01)
    denom_w = np.sqrt(vart11 + vart00)

    zc = (mut10 - mut01) / denom_c if denom_c > 0 else 0
    zw = (mut11 - mut00) / denom_w if denom_w > 0 else 0

    pc = norm.sf(np.abs(zc)) * 2
    pw = norm.sf(np.abs(zw)) * 2

    return [pc, pw], (mut10 - mut01), (mut11 - mut00)


def generate_batch_counter(df_result, n_samples):
    """Vectorized generation of comparison pairs."""
    # Oversample to account for filtering where Y1 == Y2
    batch_size = n_samples * 4

    # Use numpy values for speed
    indices = np.random.randint(0, len(df_result), (2, batch_size))

    y_vals = df_result['Y'].values
    y1 = y_vals[indices[0]]
    y2 = y_vals[indices[1]]

    # Filter for different labels
    mask = y1 != y2

    # If we don't have enough samples, just take what we have (or could loop to get more)
    valid_idx = np.where(mask)[0][:n_samples]

    idx1 = indices[0][valid_idx]
    idx2 = indices[1][valid_idx]

    c_vals = df_result['C'].values
    a_vals = df_result['A'].astype(str).values
    y_vals = df_result['Y'].values

    c1 = c_vals[idx1]
    c2 = c_vals[idx2]

    y1 = y_vals[idx1]
    y2 = y_vals[idx2]

    # Calculate C comparison code
    # x: C1==C2, 1: C1>C2, 0: C1<C2
    cij = np.select([c1 == c2, c1 > c2], ['x', '1'], default='0')
    yij = np.select([y1 < y2, y1 > y2], ['0', '1'])

    a1_str = a_vals[idx1].astype(str)
    a2_str = a_vals[idx2].astype(str)

    # Construct strings: cij + Y1 + A1 + A2
    xij = np.char.add(np.char.add(np.char.add(cij, yij), a1_str), a2_str)

    return Counter(xij)

def generate_batch_counter_sep(df_result, n_samples):
    """
    Bootstrap a sample of size n_samples from rows of df_result (with replacement),
    and count 'CYA' strings needed by separation(): '111','011','101','001','110','010','100','000'.
    """
    idx = np.random.randint(0, len(df_result), size=n_samples)
    c = df_result["C"].values[idx].astype(int).astype(str)
    y = df_result["Y"].values[idx].astype(int).astype(str)
    a = df_result["A"].values[idx].astype(int).astype(str)
    keys = np.char.add(np.char.add(c, y), a)
    return Counter(keys)


df_count_test = pd.DataFrame()
df_count_train = pd.DataFrame()
df_pred_test = pd.DataFrame()
df_pred_train = pd.DataFrame()
df_AOD_test = {}
df_EOD_test = {}
df_AOD_train = {}
df_EOD_train = {}
df_violate_test = {}
df_violate_train = {}
df_violate_sep_test = {}
df_violate_sep_train = {}
df_cross_test = {}
df_cross_train = {}
df_within_test = {}
df_within_train = {}

# For a single run (loop logic removed or uncomment for multiple)
df, df_name, train, test, sa = make_comm()
train.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True, drop=True)

y_train = train['output']

if y_train.nunique() == 2:
    isContinuous = False
else:
    isContinuous = True

train = train.drop(columns=['output'])

y_test = test['output']
test = test.drop(columns=['output'])

rng = np.random.default_rng(0)

for treatment in ['None','FairBalance', 'Reweighing']:

    if treatment == "None":
        sample_weight = None
    else:
        sample_weight = weight(train['sa'].to_numpy(), y_train.to_numpy(), treatment=treatment, isContinuous=isContinuous)

    if isContinuous:
        clf = LinearRegression().fit(train, y_train, sample_weight=sample_weight)
    else:
        clf = LogisticRegression().fit(train, y_train, sample_weight=sample_weight)

    predictions = clf.predict(test)
    predictions_train = clf.predict(train)
    df_pred_test[treatment] = predictions
    df_pred_train[treatment] = predictions_train

    df_result_test = pd.DataFrame({
        'C': predictions,
        'Y': y_test,
        'A': test['sa']
    })

    df_result_train = pd.DataFrame({
        'C': predictions_train,
        'Y': y_train,
        'A': train['sa']
    })

    count_test = df_result_test[['C', 'Y', 'A']].value_counts(normalize=True)
    count_train = df_result_train[['C', 'Y', 'A']].value_counts(normalize=True)
    df_count_test[treatment] = count_test
    df_count_train[treatment] = count_train

    if isContinuous == False:
        m_test = Metrics(y_test, predictions)
        df_AOD_test[treatment] = m_test.AOD(test['sa'])
        df_EOD_test[treatment] = m_test.EOD(test['sa'])

        m_train = Metrics(y_train, predictions_train)
        df_AOD_train[treatment] = m_train.AOD(train['sa'])
        df_EOD_train[treatment] = m_train.EOD(train['sa'])

        violate_sep_test = 0
        violate_sep_train = 0

        for _ in range(r):
            x = generate_batch_counter_sep(df_result_test, len(test))
            ps = separation(x)
            if min(ps) < alpha:
                violate_sep_test += 1

        for _ in range(r):
            x = generate_batch_counter_sep(df_result_train, len(train))
            ps = separation(x)
            if min(ps) < alpha:
                violate_sep_train += 1

        df_violate_sep_test[treatment] = violate_sep_test / r
        df_violate_sep_train[treatment] = violate_sep_train / r

    # Optimized Test Simulation
    violate_comp_test = 0
    avg_cross_test = 0
    avg_within_test = 0

    for _ in range(r):
        # Generate counter efficiently
        count = generate_batch_counter(df_result_test, 2 * len(test))

        ps, cross, within = comparative_separation(count)
        if min(ps) < alpha:
            violate_comp_test += 1

        avg_cross_test += cross
        avg_within_test += within

    # Optimized Train Simulation
    violate_comp_train = 0
    avg_cross_train = 0
    avg_within_train = 0

    for _ in range(r):
        count = generate_batch_counter(df_result_train, 2 * len(train))

        ps, cross, within = comparative_separation(count)
        if min(ps) < alpha:
            violate_comp_train += 1

        avg_cross_train += cross
        avg_within_train += within


    df_violate_test[treatment] = violate_comp_test / r
    df_cross_test[treatment] = avg_cross_test / r
    df_within_test[treatment] = avg_within_test / r

    df_violate_train[treatment] = violate_comp_train / r
    df_cross_train[treatment] = avg_cross_train / r
    df_within_train[treatment] = avg_within_train / r

if isContinuous == False:
# Save results
    df_count_test.to_csv('probability_' + df_name + '_' + sa + ".csv")
    df_count_train.to_csv('probability_' + df_name + '_' + sa + "_train.csv")

    pd.DataFrame({
        "AOD": pd.Series(df_AOD_test),
        "EOD": pd.Series(df_EOD_test),
        "Violate_Sep": pd.Series(df_violate_sep_test),
        "Violate": pd.Series(df_violate_test),
        "Cross": pd.Series(df_cross_test),
        "Within": pd.Series(df_within_test)
    }).to_csv("Result_test_" + df_name + "_" + str(2 * len(test)) + '_' + sa + ".csv")

    pd.DataFrame({
        "AOD": pd.Series(df_AOD_train),
        "EOD": pd.Series(df_EOD_train),
        "Violate_Sep": pd.Series(df_violate_sep_train),
        "Violate": pd.Series(df_violate_train),
        "Cross": pd.Series(df_cross_train),
        "Within": pd.Series(df_within_train)
    }).to_csv("Result_train_" + df_name + "_" + str(2 * len(train)) + '_' + sa + ".csv")

else:
    pd.DataFrame({
        "Violate": pd.Series(df_violate_test),
        "Cross": pd.Series(df_cross_test),
        "Within": pd.Series(df_within_test)
    }).to_csv("Result_test_" + df_name + "_" + str(2 * len(test)) + '_' + sa + ".csv")

    pd.DataFrame({
        "Violate": pd.Series(df_violate_train),
        "Cross": pd.Series(df_cross_train),
        "Within": pd.Series(df_within_train)
    }).to_csv("Result_train_" + df_name + "_" + str(2 * len(train)) + '_' + sa + ".csv")

