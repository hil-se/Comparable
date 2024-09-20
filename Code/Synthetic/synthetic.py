import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from metrics import Metrics

num_comp = 5
col = "income"

df1 = pd.DataFrame(np.array(
    [
        [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [1, 0, 0], [1, 0, 0],
    ]),
    columns=['gender', 'income', 'pred'])

df2 = pd.DataFrame(np.array(
    [
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1],
        [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [1, 0, 0], [1, 0, 0],
    ]),
    columns=['gender', 'income', 'pred'])

df3 = pd.DataFrame(np.array(
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

df = df2
# df = pd.read_csv("../../Data/adult.csv").sample(frac=0.01)
# df = df[["gender", "income"]]
#
# label_encoder = LabelEncoder()
#
# for cat in ["gender", "income"]:
#     df[cat] = label_encoder.fit_transform(df[cat])
#
# df["pred"] = df["income"]
#
# for i, row in df.iterrows():
#     rand = np.random.random()
#     pred = 1 if rand < 0.7 else 0
#     df.loc[i, "pred"] = pred

# tn, fp, fn, tp = confusion_matrix(df["income"], df["pred"]).ravel()
m = Metrics(df["income"], df["pred"])

AOD = m.AOD(df["gender"])
gAOD = m.gAOD(df["gender"])
MI = mutual_info_score(df["pred"], df["gender"])

res_tr = []
for indexA in range(0,len(df)):
    comp = []
    rowA = df.iloc[indexA]
    # while len(comp) < num_comp:
    #     indexB = random.randint(0, len(df) - 1)
    #     rowB = df.iloc[indexB]
    for indexB in range(0 , len(df)):
        # if (indexA == indexB) or (indexB in comp):
        #     continue
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
from pdb import set_trace
set_trace()
print(df)
