import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import Classification
from Code.ImageExp import DataProcessing, vgg_pre
from metrics import Metrics


def retrievePixels(path, height, width):
    # img = tf.keras.utils.load_img("../data/images/"+path, grayscale=False
    folder_path = "../../Data/Images/"
    # folder_path = "../../../XAI_Image/data/images/"
    img = tf.keras.utils.load_img(folder_path + path, target_size=(height, width))
    x = tf.keras.utils.img_to_array(img)
    return x


num_comp_train = 1
num_comp_test = 1

col = "output"


# 1,2,3,4
def make_df1():
    df1 = pd.DataFrame(np.array(
        [
            [0, 1, 1],
            [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [1, 1, 1], [1, 1, 1],
            [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 1, 1],
            [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [1, 1, 1], [1, 1, 1],
            [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 1, 1],
            [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [1, 1, 1], [1, 1, 1],
            [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        ]),
        columns=['gender', 'income', 'pred'])

    return df1, "df1"


# 4,2,3,1
def make_df2():
    df2 = pd.DataFrame(np.array(
        100 * [
            [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1],
            [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
            [0, 0, 0],
            [1, 1, 1], [1, 1, 1],
            [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1],
            [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
            [0, 0, 0],
            [1, 1, 1], [1, 1, 1],
            [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1],
            [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
            [0, 0, 0],
            [1, 1, 1], [1, 1, 1],
            [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        ]),
        columns=['gender', 'income', 'pred'])

    return df2, "df2"


# 1,4,3,8
def make_df3():
    df3 = pd.DataFrame(np.array(
        [
            [0, 1, 1],
            [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [1, 1, 1], [1, 1, 1],
            [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 1, 1],
            [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [1, 1, 1], [1, 1, 1],
            [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 1, 1],
            [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [1, 1, 1], [1, 1, 1],
            [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        ]),
        columns=['gender', 'income', 'pred'])
    return df3, "df3"


def make_df4():
    df4 = pd.DataFrame(np.array(
        [
            [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 1, 0],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 1, 0],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 1, 0],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 1, 0],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 1, 0],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],

        ]),
        columns=['gender', 'income', 'pred'])
    return df4, "df4"


def make_df5():
    df5 = pd.DataFrame(np.array(
        [
            [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 1, 0],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 1, 0],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 1, 0],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 1, 0],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
            [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [1, 1, 0],
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],

        ]),
        columns=['gender', 'income', 'pred'])
    return df5, "df5"


def make_df6(n=1000, p1=0.5, p2=0.5, p3=0.5):
    # n is the number of data points.
    # 0 <= p <= 1 is the sampling probability of Male (sex=1).
    keys = ['gender', 'income', 'pred']
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


def make_scut(P="P3"):
    df = pd.read_csv("../../Data/ImageExp/Selected_Ratings.csv")
    df = df[["Filename", P]]
    df['pixels'] = df['Filename'].apply(DataProcessing.retrievePixels)

    train_val = df.sample(frac=0.8)
    test = df.drop(train_val.index)
    train = train_val.sample(frac=0.8)
    val = train_val.drop(train.index)

    features_ts = np.array([pixel for pixel in test['pixels']]) / 255.0
    features_tr = np.array([pixel for pixel in train['pixels']]) / 255.0
    features_val = np.array([pixel for pixel in val['pixels']]) / 255.0

    X_train = features_tr
    X_val = features_val
    X_test = features_ts

    model = vgg_pre.VGG_Pre()

    model.fit(X_train, train[P], X_val, val[P])

    protected_ts_sex = []
    for file in test["Filename"]:
        if file[1] == 'M':
            protected_ts_sex.append(1)
        else:
            protected_ts_sex.append(0)

    test['gender'] = protected_ts_sex

    pred = model.decision_function(X_test)
    pred = (pred >= 3).astype(int)
    test['income'] = (test[P] >= 3).astype(int)
    test['pred'] = pred
    test.reset_index(inplace=True, drop=True)

    df = test[["income", "gender", "pred"]]
    return df, "scut" + "_" + str(P)


def make_adult():
    seed = 18
    df = pd.read_csv("../../Data/adult.csv", na_values=["?"])
    df = df.dropna()
    df['gender'] = df['gender'].apply(lambda x: 1 if x == "Male" else 0)
    df['income'] = df['income'].apply(lambda x: 1 if x == ">50K" else 0)
    dependent = 'income'

    sa = 'gender'

    df = df.rename(columns={sa: 'sa'})

    df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation',
                                     'relationship', 'race', 'native-country'], dtype=float,
                        drop_first=True)

    X = df.drop([dependent], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)

    clf = LogisticRegression(random_state=seed).fit(X_train, y_train)

    predictions = clf.predict(X_test)

    index_pos = 1 - clf.classes_[0]
    pred_prob = clf.predict_proba(X_test)[:, index_pos]

    accuracy = accuracy_score(y_test, predictions)

    X_test_cp = X_test.copy()
    X_test_cp['sa'] = X_test['sa']
    X_test_cp['pred_con'] = pred_prob
    X_test_cp['pred'] = predictions
    X_test_cp[col] = y_test
    X_test_cp.reset_index(inplace=True, drop=True)

    X_train[col] = y_train
    X_test[col] = y_test

    df = X_test_cp[[col, "sa", "pred", "pred_con"]]
    return df, "adult", X_train, X_test


def make_german():
    seed = 42
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
        X, y, test_size=0.2, random_state=seed)

    clf = LogisticRegression(random_state=seed).fit(X_train, y_train)

    predictions = clf.predict(X_test)

    index_pos = 1 - clf.classes_[0]
    pred_prob = clf.predict_proba(X_test)[:, index_pos]

    accuracy = accuracy_score(y_test, predictions)

    X_test_cp = X_test.copy()
    X_test_cp['sa'] = X_test['sa']
    X_test_cp['pred_con'] = pred_prob
    X_test_cp['pred'] = predictions
    X_test_cp[col] = y_test
    X_test_cp.reset_index(inplace=True, drop=True)

    X_train[col] = y_train
    X_test[col] = y_test

    df = X_test_cp[[col, "sa", "pred", "pred_con"]]
    return df, "german", X_train, X_test


def make_heart():
    seed = 42
    df = pd.read_csv("../../Data/heart.csv")
    df = df.dropna()

    dependent = 'output'
    sa = 'sex'

    df = df.rename(columns={sa: 'sa'})

    X = df.drop([dependent], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)

    clf = LogisticRegression(random_state=seed).fit(X_train, y_train)

    predictions = clf.predict(X_test)

    index_pos = 1 - clf.classes_[0]
    pred_prob = clf.predict_proba(X_test)[:, index_pos]

    accuracy = accuracy_score(y_test, predictions)

    X_test_cp = X_test.copy()
    X_test_cp['sa'] = X_test['sa']
    X_test_cp['pred_con'] = pred_prob
    X_test_cp['pred'] = predictions
    X_test_cp[col] = y_test
    X_test_cp.reset_index(inplace=True, drop=True)

    X_train[col] = y_train
    X_test[col] = y_test

    df = X_test_cp[[col, "sa", "pred", "pred_con"]]
    return df, "heart", X_train, X_test


results = []
df, df_name, train, test = make_adult()
train.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True, drop=True)

res_tr_encoder = []

for indexA, rowA in train.iterrows():
    comp = []
    train_cp = train.copy()
    comp_count = 0
    while comp_count < num_comp_train:
        rowB = train_cp.sample()
        indexB = rowB.index[0]
        if (indexB == indexA):
            continue
        rowB = rowB.iloc[0]
        ratingA = rowA[col]
        ratingB = rowB[col]
        label = 0
        if ratingA > ratingB:
            label = 1
        elif ratingA < ratingB:
            label = -1
        if label != 0:
            # if label is not None:
            trainA = rowA.drop(labels=[col])
            trainB = rowB.drop(labels=[col])

            res_tr_encoder.append({"A": trainA.to_list(),
                                   "B": trainB.to_list(),
                                   "Label": label
                                   })
            train_cp.drop(indexB, inplace=True)
            comp_count += 1

data_tr_encoder = pd.DataFrame(res_tr_encoder)

train_encoder = data_tr_encoder.sample(frac=0.85)
y_true = train_encoder["Label"].tolist()
val = data_tr_encoder.drop(train_encoder.index)

dual_encoder = Classification.train_model(train=train_encoder, val=val, y_true=y_true, shared=True, epochs=500)

for i in range(10):

    # m = Metrics(df["income"], df["pred"])
    # AOD = m.AOD(df["gender"])
    # EOD = m.EOD(df["gender"])
    # gAOD = m.gAOD(df["gender"])
    # MI = m.MI_b(df["gender"])

    res_ts_encoder = []
    test_list = []

    for indexA, rowA in test.iterrows():
        comp = []
        test_cp = test.copy()
        comp_count = 0
        while comp_count < num_comp_test:
            rowB = test_cp.sample()
            indexB = rowB.index[0]
            if (indexB == indexA):
                continue
            rowB = rowB.iloc[0]
            ratingA = rowA[col]
            ratingB = rowB[col]
            label = 0
            if ratingA > ratingB:
                label = 1
            elif ratingA < ratingB:
                label = -1
            if label != 0:
                # if label is not None:
                testA = rowA.drop(labels=[col])
                testB = rowB.drop(labels=[col])

                res_ts_encoder.append({"A": testA.to_list(),
                                       "B": testB.to_list(),
                                       "Label": label
                                       })
                test_list.append({"A": testA['sa'],
                                  "B": testB['sa'],
                                  "Label": label
                                  })
                test_cp.drop(indexB, inplace=True)
                comp_count += 1

    data_ts_encoder = pd.DataFrame(res_ts_encoder)
    test_list = pd.DataFrame(test_list)

    predictions = Classification.predict(data_ts_encoder, dual_encoder)

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

    test_list["pred"] = predictions
    m = Metrics(test_list["Label"], test_list["pred"])
    AOD_comp = m.AOD_comp(test_list[["A", "B"]])
    Within_comp = m.Within_comp(test_list[["A", "B"]])
    Sep_comp = m.Sep_comp(test_list[["A", "B"]])
# MI_comp = m.MI_comp(data_tr[["A", "B"]])
# MI_comp2 = m.MI_comp2(data_tr[["A", "B"]])

    result = {"# of comparisons": len(test_list), "AOD_comp": AOD_comp,
              "Within_comp": Within_comp, "EOD_comp": AOD_comp + Within_comp,
              # "MI_comp": MI_comp, "MI_comp2": MI_comp2, "Ratio": MI / MI_comp
              }
    results.append(result)

results = pd.DataFrame(results)
results.loc[len(results.index)] = results.mean()
results.loc[len(results.index)] = results.std()
results.to_csv(df_name + "_encoder_" + str(num_comp_train) + '_' + str(num_comp_test) +".csv", index=False)

# experiment with the num of comparison (repeat 20 times and get mean and std)
# repeated trail on df1-df3 and add more data points
# SCUT dataset
# Find out the relationship between num of comparison and num of data points
