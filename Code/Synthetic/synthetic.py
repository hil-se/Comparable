import random

import numpy as np
import pandas as pd
import tensorflow as tf

from Code.ImageExp import DataProcessing, vgg_pre
from Code.ImageExp.DataProcessing import processData
from metrics import Metrics

def retrievePixels(path, height, width):
    # img = tf.keras.utils.load_img("../data/images/"+path, grayscale=False)
    folder_path = "../../Data/Images/"
    # folder_path = "../../../XAI_Image/data/images/"
    img = tf.keras.utils.load_img(folder_path + path, target_size=(height, width))
    x = tf.keras.utils.img_to_array(img)
    return x

num_comp = 1
col = "income"

# 1,2,3,4
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
# 4,2,3,1
df2 = pd.DataFrame(np.array(
    100*[
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
# 1,4,3,8
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

df4 = pd.DataFrame(np.array(
    [
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],

    ]),
    columns=['gender', 'income', 'pred'])

df5 = pd.DataFrame(np.array(
    [
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],

    ]),
    columns=['gender', 'income', 'pred'])

def make_df5(n=1000, p1=0.5, p2=0.5, p3=0.5):
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
    df = pd.DataFrame(data, columns = keys)
    return df

df6 = pd.read_csv("../../Data/ImageExp/Selected_Ratings.csv")
df6 = df6[["Filename", "P1"]]
df6['pixels'] = df6['Filename'].apply(DataProcessing.retrievePixels)

train_val = df6.sample(frac=0.8)
test = df6.drop(train_val.index)
train = train_val.sample(frac=0.8)
val = train_val.drop(train.index)

features_ts = np.array([pixel for pixel in test['pixels']]) / 255.0
features_tr = np.array([pixel for pixel in train['pixels']]) / 255.0
features_val = np.array([pixel for pixel in val['pixels']]) / 255.0

X_train = features_tr
X_val = features_val
X_test = features_ts

model = vgg_pre.VGG_Pre()

model.fit(X_train, train["P1"], X_val, val["P1"])

protected_ts_sex = []
for file in test["Filename"]:
    if file[1] == 'M':
        protected_ts_sex.append(1)
    else:
        protected_ts_sex.append(0)

test['gender'] = protected_ts_sex

pred = model.decision_function(X_test)
pred = (pred >= 3).astype(int)
test['income'] = (test["P1"] >= 3).astype(int)
test['pred'] = pred
test.reset_index(inplace=True, drop=True)

df = test[["income","gender","pred"]]

m = Metrics(df["income"], df["pred"])
AOD = m.AOD(df["gender"])
gAOD = m.gAOD(df["gender"])
MI = m.MI_b(df["gender"])

results = []

for i in range(20):
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
results.to_csv("scut_" + str(len(comp)) + ".csv", index=False)

# experiment with the num of comparison (repeat 20 times and get mean and std)
# repeated trail on df1-df3 and add more data points
# SCUT dataset
# Find out the relationship between num of comparison and num of data points