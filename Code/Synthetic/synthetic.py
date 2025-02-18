import random
from os import access

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.metrics import accuracy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from Code.ImageExp import DataProcessing, vgg_pre
from metrics import Metrics
from sklearn.linear_model import LogisticRegression



def retrievePixels(path, height, width):
    # img = tf.keras.utils.load_img("../data/images/"+path, grayscale=False)
    folder_path = "../../Data/Images/"
    # folder_path = "../../../XAI_Image/data/images/"
    img = tf.keras.utils.load_img(folder_path + path, target_size=(height, width))
    x = tf.keras.utils.img_to_array(img)
    return x


num_comp = 10
col = "income"


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
    df = pd.read_csv("../../Data/adult.csv", na_values=["?"])
    df = df.dropna()
    df['gender'] = df['gender'].apply(lambda x: 1 if x == "Male" else 0)
    df['income'] = df['income'].apply(lambda x: 1 if x == ">50K" else 0)
    dependent = 'income'

    df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation',
                                                    'relationship', 'race', 'native-country'], dtype=float,
                                       drop_first=True)

    X = df.drop([dependent], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    clf = LogisticRegression().fit(X_train, y_train)

    predictions = clf.predict(X_test)

    X_test['pred'] = predictions
    X_test['income'] = y_test
    X_test.reset_index(inplace=True, drop=True)

    df = X_test[["income", "gender", "pred"]]
    return df, "adult"

def make_german():
    df = pd.read_csv("../../Data/german_credit_data.csv", index_col=0)
    df = df.dropna()
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == "male" else 0)
    df['Risk'] = df['Risk'].apply(lambda x: 1 if x == "good" else 0)

    dependent = 'Risk'

    df = pd.get_dummies(df, columns=['Housing', 'Saving accounts', 'Checking account', 'Purpose'], dtype=float,
                                       drop_first=True)

    X = df.drop([dependent], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    clf = LogisticRegression().fit(X_train, y_train)

    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    X_test['gender'] = X_test['Sex']
    X_test['pred'] = predictions
    X_test['income'] = y_test
    X_test.reset_index(inplace=True, drop=True)

    df = X_test[["income", "gender", "pred"]]
    return df, "german"

def make_heart():
    df = pd.read_csv("../../Data/heart.csv")
    df = df.dropna()

    dependent = 'output'

    X = df.drop([dependent], axis=1)
    y = np.array(df[dependent])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    clf = LogisticRegression().fit(X_train, y_train)

    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    X_test['gender'] = X_test['sex']
    X_test['pred'] = predictions
    X_test['income'] = y_test
    X_test.reset_index(inplace=True, drop=True)

    df = X_test[["income", "gender", "pred"]]
    return df, "heart"


results = []

for i in range(10):
    df, df_name = make_heart()

    m = Metrics(df["income"], df["pred"])
    AOD = m.AOD(df["gender"])
    gAOD = m.gAOD(df["gender"])
    MI = m.MI_b(df["gender"])

    res_tr = []
    for indexA in range(0, len(df)):
        comp = []
        rowA = df.iloc[indexA]
        while len(comp) < num_comp:
            indexB = random.randint(0, len(df) - 1)
            # for indexB in range(0, len(df)):
            #     if (indexB in comp):
            #         continue
            rowB = df.iloc[indexB]
            ratingA = rowA[col]
            ratingB = rowB[col]
            predA = rowA["pred"]
            predB = rowB["pred"]
            label = 0
            pred = 0
            if ratingA > ratingB:
                label = 1
            elif ratingA < ratingB:
                label = -1
            if predA > predB:
                pred = 1
            elif predA < predB:
                pred = -1
            res_tr.append({"A": rowA["gender"],
                           "B": rowB["gender"],
                           "Label": label,
                           "pred": pred
                           })
            comp.append(indexB)
    data_tr = pd.DataFrame(res_tr)

    m = Metrics(data_tr["Label"], data_tr["pred"])
    AOD_comp = m.AOD_comp(data_tr[["A", "B"]])
    Within_comp = m.Within_comp(data_tr[["A", "B"]])
    Sep_comp = m.Sep_comp(data_tr[["A", "B"]])
    # MI_comp = m.MI_comp(data_tr[["A", "B"]])
    # MI_comp2 = m.MI_comp2(data_tr[["A", "B"]])

    result = {"AOD": AOD, "gAOD": gAOD,
              "MI": MI, "AOD_comp": AOD_comp, "Within_comp": Within_comp, "Sep_comp": Sep_comp,
              # "MI_comp": MI_comp, "MI_comp2": MI_comp2, "Ratio": MI / MI_comp
              }
    results.append(result)

results = pd.DataFrame(results)
results.loc[len(results.index)] = results.mean()
results.loc[len(results.index)] = results.std()
results.to_csv(df_name + "_" + str(num_comp) + ".csv", index=False)

# experiment with the num of comparison (repeat 20 times and get mean and std)
# repeated trail on df1-df3 and add more data points
# SCUT dataset
# Find out the relationship between num of comparison and num of data points
