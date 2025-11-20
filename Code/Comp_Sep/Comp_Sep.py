from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.pyplot import plot
from numpy import linspace
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.svm import LinearSVC
from scipy.stats import norm, bernoulli

from Code.ImageExp import DataProcessing, vgg_pre
from metrics import Metrics

# split the dataset in half stratifily on SA
# train a classification model and one with fairbalance
# sample without replacement for a certain number of repetition

col = "output"

alpha = 0.05
r = 100
nc = 2000


def make_german():
    # seed = 42
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

    return df, "german", X_train, X_test


def make_heart():
    # seed = 42
    df = pd.read_csv("../../Data/heart.csv")
    df = df.dropna()

    dependent = 'output'
    sa = 'sex'

    df = df.rename(columns={sa: 'sa'})

    X = df.drop([dependent], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5)


    X_train[col] = y_train
    X_test[col] = y_test

    return df, "heart", X_train, X_test


def make_adult():
    df = pd.read_csv("../../Data/adult.csv", na_values=["?"])
    # df = df.sample(frac=0.1)
    df = df.dropna()
    df['gender'] = df['gender'].apply(lambda x: 1 if x == "Male" else 0)
    df['income'] = df['income'].apply(lambda x: 1 if x == ">50K" else 0)
    dependent = 'income'

    sa = 'gender'

    df = df.rename(columns={sa: 'sa'})

    df = pd.get_dummies(df, columns=['workclass', 'marital-status', 'occupation',
                                     'relationship', 'race'], dtype=float,
                        drop_first=True)

    X = df.drop([dependent, 'education', 'native-country'], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=df['sa'])

    X_train[col] = y_train
    X_test[col] = y_test

    return df, "adult", X_train, X_test


def density(X):
    w = []
    X = X.tolist()
    df = pd.DataFrame(X)
    for index, row in df.iterrows():
        row = list(row)
        if len(row) == 2:
            w.append(X.count(row) / len(X))
        else:
            w.append(X.count(row[0]) / len(X))
    return np.array(w)


def weight(A, y, treatment="FairBalance"):
    X = np.stack((A, y), axis=1)

    w = density(X)
    wA = density(A)
    wy = density(y)

    if treatment == "FairBalanceVariant":
        weight = 1 / w
    elif treatment == "FairBalance":
        weight = wA / w
    elif treatment == "GroupBalance":
        weight = wy / w
    elif treatment == "Reweighing":
        weight = wA * wy / w

    weight = (len(weight) * weight / sum(weight)).flatten()
    return weight


def cal_comp(xt1, x1, xt2, x2):
    mu = (xt1 + xt2) / (x1 + x2)
    var = mu * (1 - mu) / (x1 + x2)
    return mu, var


def comparative_separation(x):
    mut11, vart11 = cal_comp(x["1111"], x["1111"] + x["0111"] + x["x111"], x["0011"], x["0011"] + x["1011"] + x["x011"])
    mut00, vart00 = cal_comp(x["1100"], x["1100"] + x["0100"] + x["x100"], x["0000"], x["0000"] + x["1000"] + x["x000"])
    mut10, vart10 = cal_comp(x["1110"], x["1110"] + x["0110"] + x["x110"], x["0001"], x["0001"] + x["1001"] + x["x001"])
    mut01, vart01 = cal_comp(x["1101"], x["1101"] + x["0101"] + x["x101"], x["0010"], x["0010"] + x["1010"] + x["x010"])
    zc = (mut10 - mut01) / np.sqrt(vart10 + vart01)
    zw = (mut11 - mut00) / np.sqrt(vart11 + vart00)
    pc = norm.sf(np.abs(zc)) * 2
    pw = norm.sf(np.abs(zw)) * 2

    return [pc, pw], (mut10 - mut01), (mut11 - mut00)


results = []
df_count = pd.DataFrame()
df_pred = pd.DataFrame()
df_AOD = pd.Series()
df_EOD = pd.Series()
df_violate = pd.Series()
df_cross = pd.Series()
df_within = pd.Series()

# for i in range(10):
df, df_name, train, test = make_german()
train.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True, drop=True)

y_train = train['output']
train = train.drop(columns=['output'])

y_test = test['output']
test = test.drop(columns=['output'])

for treatment in ['None', 'FairBalance', 'Reweighing']:

    if treatment == "None":
        sample_weight = None
    else:
        sample_weight = weight(train['sa'].to_numpy(), y_train.to_numpy(), treatment=treatment)

    clf = LogisticRegression().fit(train, y_train, sample_weight=sample_weight)
    predictions = clf.predict(test)
    df_pred[treatment] = predictions

    df_result = np.stack((predictions, y_test, test['sa']), axis=1)
    df_result = pd.DataFrame(df_result, columns=['C', 'Y', 'A'])

    count = df_result.value_counts(normalize=True)
    df_count[treatment] = count

    m = Metrics(y_test, predictions)
    df_AOD[treatment] = m.AOD(test['sa'])
    df_EOD[treatment] = m.EOD(test['sa'])

    violate = 0
    avg_cross = 0
    avg_within = 0

    for i in range(r):
        x = []
        j = 0
        while j < nc:
            c1 = df_result.sample()
            index_c1 = c1.index[0]
            c2 = df_result.sample()
            index_c2 = c2.index[0]
            if c1['Y'].item() == c2['Y'].item():
                continue
            if c1['C'].item() == c2['C'].item():
                cij = "x"
            elif c1['C'].item() > c2['C'].item():
                cij = "1"
            else:
                cij = "0"
            aij = str(c1['A'].item()) + str(c2['A'].item())
            xij = cij + str(c1['Y'].item()) + aij
            x.append(xij)
            j = j + 1

        count = Counter(x)

        ps, cross, within = comparative_separation(count)
        if min((ps)) < alpha:
            violate += 1

        avg_cross = avg_cross + cross
        avg_within = avg_within + within

    violate_r = (violate / r)
    avg_cross_r = (avg_cross / r)
    avg_within_r = (avg_within / r)

    df_violate[treatment] = violate_r
    df_cross[treatment] = avg_cross_r
    df_within[treatment] = avg_within_r

df_count.to_csv('probability_' + df_name + ".csv")
df_AOD.to_csv('AOD_' + df_name + ".csv")
df_EOD.to_csv('EOD_' + df_name + ".csv")
df_violate.to_csv('violate_' + df_name + '_' + str(nc) + ".csv")
df_cross.to_csv('cross_' + df_name + '_' + str(nc) + ".csv")
df_within.to_csv('within_' + df_name + '_' + str(nc) + ".csv")
