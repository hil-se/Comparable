import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from metrics import Metrics

seed = 18

df = pd.read_csv("../../Data/adult.csv", na_values=["?"])
df = df.dropna()
df['gender'] = df['gender'].apply(lambda x: 1 if x == "Male" else 0)
df['income'] = df['income'].apply(lambda x: 1 if x == ">50K" else 0)
df['race'] = df['race'].apply(lambda x: 1 if x == "White" else 0)

dependent = 'income'

df = pd.get_dummies(df, columns=['workclass', 'marital-status', 'occupation',
                                 'relationship'], dtype=float,
                    drop_first=True)

X = df.drop([dependent, 'education', 'native-country'], axis=1)
y = np.array(df[dependent])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed)

# clf = LogisticRegression(random_state=seed).fit(X_train, y_train)
#
# predictions = clf.predict(X_test)
#
# predictions_all = clf.predict(X)
#
# accuracy = accuracy_score(y_test, predictions)

# m_all = Metrics(y_test,predictions)
# AOD_all = m_all.AOD(X_test['gender'])
# EOD_all = m_all.EOD(X_test['gender'])

# m_all = Metrics(y, predictions_all)
# AOD_all = m_all.AOD(X['gender'])
# EOD_all = m_all.EOD(X['gender'])

X['target'] = y
# X['pred'] = predictions_all

# results = []
# results_sample = []
# results_comp = []
#
# result_all = {"# of comparisons": "ALL", "AOD": AOD_all, "EOD": EOD_all}

# sample_size = 3000
num_comp_test = 10000

for i in range(20):
    # X_test_sample, y_test_sample = resample(X_test, y_test, n_samples=sample_size, replace=False)
    # predictions_sample = clf.predict(X_test_sample)
    #
    # accuracy_sample = accuracy_score(y_test_sample, predictions_sample)
    #
    # m_sample = Metrics(y_test_sample,predictions_sample)
    # AOD_sample = m_sample.AOD(X_test_sample['gender'])
    # EOD_sample = m_sample.EOD(X_test_sample['gender'])
    #
    # result = {"# of comparisons": len(X_test_sample), "AOD": AOD_sample, "EOD": EOD_sample}
    # results_sample.append(result)
    #
    # X_test_sample['target'] = y_test_sample
    # X_test_sample['pred'] = predictions_sample

    res_ts_encoder = []
    test_list = []

    comp_count = 0

    while comp_count < num_comp_test:
        rowA = X.sample()
        indexA = rowA.index[0]
        comp = []
        test_cp = X.copy()
        rowB = test_cp.sample()
        indexB = rowB.index[0]

        if (indexB == indexA):
            continue

        rowB = rowB.iloc[0]
        rowA = rowA.iloc[0]
        ratingA = rowA['target']
        ratingB = rowB['target']
        predA = rowA['pred']
        predB = rowB['pred']
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

        if label != 0:
            testA = rowA.drop(labels=['target'])
            testB = rowB.drop(labels=['target'])
            # res_ts_encoder.append({"A": testA.to_list(),
            #                        "B": testB.to_list(),
            #                        "Label": label
            #                        })
            test_list.append({"A": testA['gender'],
                              "B": testB['gender'],
                              "Label": label,
                              "Pred": pred,
                              })
            test_cp.drop(indexB, inplace=True)
            comp_count += 1

    data_ts_encoder = pd.DataFrame(res_ts_encoder)
    test_list = pd.DataFrame(test_list)

    m_comp = Metrics(test_list["Label"], test_list["Pred"])
    AOD_comp = m_comp.AOD_comp(test_list[["A", "B"]])
    Within_comp = m_comp.Within_comp(test_list[["A", "B"]])
    Sep_comp = m_comp.Sep_comp(test_list[["A", "B"]])

    result_comp = {"# of comparisons": "COMP", "AOD": AOD_comp, "EOD": AOD_comp + Within_comp}
    results_comp.append(result_comp)

# results_sample = pd.DataFrame(results_sample)
results_comp = pd.DataFrame(results_comp)

# results_sample.loc[len(results_sample)] = ["AVG", results_sample['AOD'].mean(), results_sample["EOD"].mean()]
# results_sample.loc[len(results_sample)] = ["STD", results_sample['AOD'].std(), results_sample["EOD"].std()]
results_comp.loc[len(results_comp)] = ["AVG", results_comp['AOD'].mean(), results_comp["EOD"].mean()]
results_comp.loc[len(results_comp)] = ["STD", results_comp['AOD'].std(), results_comp["EOD"].std()]

# results_sample.loc[len(results_sample)] = result_all
results_comp.loc[len(results_comp)] = result_all

# results = pd.concat([results_sample, results_comp], axis=1)
results_comp.to_csv("AOD_adult_" + str(num_comp_test) + ".csv", index=False)
