import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

import DualEncoder
import SharedDualEncoder
import metrics


def _predict_labels_batch(test, dual_encoder, batch_size=2048):
    """Batch inference to avoid per-row Python/TensorFlow overhead."""
    dataA = np.asarray(test["A"].tolist())
    dataB = np.asarray(test["B"].tolist())
    raw_scores = dual_encoder.predict(dataA, dataB, batch_size=batch_size).numpy().reshape(-1)
    return np.where(raw_scores < 0, -1, 1).tolist()


def learn(train_data,
          epochs=100,
          validation_data=None,
          y_true=[],
          patience=10,
          batch_size=512,
          shared=False,
          train_weights=None,
          val_weights=None,
          df_name= None):
    # Vectorized data extraction (avoid list comprehension)
    source = np.array(train_data["A"].tolist())
    target = np.array(train_data["B"].tolist())
    train_y = train_data["Label"].values

    if train_weights is not None:
        tr_feature = {"A": source, "B": target, "Label": train_y, "Weights": train_weights}
    else:
        tr_feature = {"A": source, "B": target, "Label": train_y}

    train_dataset = tf.data.Dataset.from_tensor_slices(tr_feature)

    train_dataset = train_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    if shared == True:
        encoder = SharedDualEncoder.create_encoder(input_size=train_dataset.element_spec['A'].shape[1], df_name=df_name)
        dual_encoder = SharedDualEncoder.DualEncoderAll(encoder, y_true=np.array(y_true))
    else:
        encoder_A = DualEncoder.create_encoder(input_size=train_dataset.element_spec['A'].shape[1])
        encoder_B = DualEncoder.create_encoder(input_size=train_dataset.element_spec['A'].shape[1])
        dual_encoder = DualEncoder.DualEncoderAll(encoder_A, encoder_B, y_true=np.array(y_true))
    dual_encoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        jit_compile=True
    )
    # dual_encoder.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.001))
    dual_encoder.fit(
        x=train_dataset,
        epochs=epochs,
        verbose=0)

    return dual_encoder


def train_model(train, val, y_true, epochs=100, shared=False, train_weights=None, val_weights=None, df_name= None):
    train = train.sample(frac=1).reset_index(drop=True)
    dual_encoder = learn(train, epochs=epochs, validation_data=None, y_true=y_true, shared=shared,
                         train_weights=train_weights, val_weights=None, df_name=df_name)
    return dual_encoder


def predict(test, dual_encoder):
    return _predict_labels_batch(test, dual_encoder)


def test_model(test, dual_encoder):
    labels = test["Label"].tolist()
    predictions = _predict_labels_batch(test, dual_encoder)
    return predictions, evaluate(labels, predictions)
    # return evaluate_accuracy(labels, predictions)


def evaluate_accuracy(y_true, y_pred):
    matches = 0
    ln = len(y_true)
    for i in range(ln):
        if y_pred[i] == y_true[i]:
            matches += 1
    accuracy = matches / ln
    return accuracy


def evaluate(y_true, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    recall = 0
    precision = 0
    F1 = 0
    accuracy = 0
    ln = len(y_true)
    for i in range(ln):
        label = y_true[i]
        prediction = y_pred[i]
        if label == 1:
            if prediction == 1:
                TP += 1
            else:
                FN += 1
        else:
            if prediction == 1:
                FP += 1
            else:
                TN += 1
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    if (recall + precision) != 0:
        F1 = (2 * recall * precision) / (recall + precision)
    if (TP + FP + TN + FN) != 0:
        accuracy = (TP + TN) / (TP + FP + TN + FN)
    return recall, precision, F1, accuracy


# def generateLists(test_dataset, dual_encoder):
#     # realList = {}
#     # predList = {}
#     realList = []
#     predList = []
#     for index, row in test_dataset.iterrows():
#         idName = row["indexA"]
#         datapoint = np.array(row["A"])
#         real_score = row["Score"]
#         datapoint = np.expand_dims(datapoint, axis=0)
#         pred_score = dual_encoder.score(datapoint)
#         pred_score = pred_score.numpy()[0][0].item()
#         # realList[idName] = real_score
#         # predList[idName] = pred_score
#         realList.append({"id": idName, "Score": real_score})
#         predList.append({"id": idName, "Score": pred_score})
#     realList = pd.DataFrame(realList)
#     predList = pd.DataFrame(predList)
#     realList.sort_values(by=['Score'], inplace=True)
#     predList.sort_values(by=['Score'], inplace=True)
#     realList = realList.reset_index()
#     predList = predList.reset_index()
#     return realList, predList

def generateLists(test_dataset, dual_encoder):
    # realList = {}
    # predList = {}
    realList = []
    predList = []
    for index, row in test_dataset.iterrows():
        idName = row["indexA"]
        datapoint = np.array(row["A"])
        real_score = row["Score"]
        datapoint = np.expand_dims(datapoint, axis=0)
        pred_score = dual_encoder.score(datapoint)
        pred_score = pred_score.numpy()[0][0].item()
        # realList[idName] = real_score
        # predList[idName] = pred_score
        realList.append({"id": idName, "Score": real_score})
        predList.append({"id": idName, "Score": pred_score})
    realList = pd.DataFrame(realList)
    predList = pd.DataFrame(predList)

    m_comp = metrics.Metrics(realList["Score"], predList["Score"])

    [spearmanr, sp_pvalue] = stats.spearmanr(realList["Score"], predList["Score"])
    [pearsonr, p_pvalue] = stats.pearsonr(realList["Score"], predList["Score"])

    # realList.sort_values(by=['Score'], inplace=True)
    # predList.sort_values(by=['Score'], inplace=True)
    # realList = realList.reset_index()
    # predList = predList.reset_index()
    return spearmanr, sp_pvalue, pearsonr, p_pvalue


def evaluateLists(realList, predList):
    realList = realList["id"].tolist()
    predList = predList["id"].tolist()
    ln = len(realList)
    diff = 0
    sum_d = 0
    for i in range(ln):
        id = realList[i]
        j = predList.index(id)
        diff += (abs(i - j))
        sum_d += ((i - j) * (i - j))
    spearman_corr = 1 - ((6 * sum_d) / (ln * ((ln * ln) - 1)))
    spearman_corr = round(spearman_corr, 3)
    avg_diff = round(diff / ln, 3)
    return avg_diff, spearman_corr


def explainability(test_dataset, feat_list, dual_encoder):
    res = []
    print("\n\n")
    for index, row in test_dataset.iterrows():
        idName = row["indexA"]
        datapoint = np.array(row["A"])
        real_score = row["Score"]
        datapoint = np.expand_dims(datapoint, axis=0)
        grad = dual_encoder.output_grad(tf.Variable(datapoint))
        pred_score = dual_encoder.score(datapoint)
        pred_score = pred_score.numpy()[0][0].item()

        t_real = {"id": idName}
        t_real["Type"] = "Real (Original data)"
        for i in range(len(feat_list)):
            t_real[feat_list[i]] = (row["A"])[i]
        t_real["Score"] = real_score
        res.append(t_real)

        t_weights = {"id": idName}
        t_weights["Type"] = "Weighted features"
        weighted_feats = (np.multiply(np.array(row["A"]), grad)).tolist()
        for i in range(len(feat_list)):
            t_weights[feat_list[i]] = weighted_feats[i]
        t_weights["Score"] = pred_score
        res.append(t_weights)
        res.append({})
    res = pd.DataFrame(res)
    return res

# def comparabilityExperiment(shared=False, dataName="Boston", testList=None, dataList=None, feat_list=None, epochs=100):
#     r = Reader()
#     r.load(dataName+"_dual_train")
#     train_val = pd.concat([r.A_series, r.B_series, r.labels], axis=1)
#     np.random.shuffle(train_val.values)
#     train = train_val.head(int((len(train_val.index) * 0.7)))
#     y_true = train["Label"].tolist()
#     val = train_val.drop(train.index)
#
#     r = Reader()
#     r.load(dataName+"_dual_test")
#     test = pd.concat([r.A_series, r.B_series, r.labels], axis=1)
#     np.random.shuffle(test.values)
#
#     print("Training...")
#     dual_encoder = train_model(train=train, val=val, y_true=y_true, shared=shared, epochs=epochs)
#     print("Finished training.")
#     print("Testing...")
#
#     # recall, precision, F1, accuracy = test_model(test, dual_encoder)
#     # print(recall, precision, F1, accuracy)
#
#     accuracy = test_model(test, dual_encoder)
#     print(accuracy)
#
#     realList, predList = generateLists(testList, dual_encoder)
#     realList.to_csv("../../Results/Real Order "+dataName+".csv", index=False)
#     predList.to_csv("../../Results/Prediction Order " + dataName + ".csv", index=False)
#     avg_diff, spearman_corr = evaluateLists(realList, predList)
#     print(avg_diff, spearman_corr)
#
#     realList, predList = generateLists(dataList, dual_encoder)
#     realList.to_csv("../../Results/Real Order " + dataName + " Full.csv", index=False)
#     predList.to_csv("../../Results/Prediction Order " + dataName + " Full.csv", index=False)
#     avg_diff_full, spearman_corr_full = evaluateLists(realList, predList)
#     print(avg_diff_full, spearman_corr_full)
#
#     # expln_df = explainability(test_dataset=testList, feat_list=feat_list, dual_encoder=dual_encoder)
#     # expln_df.to_csv("../../Results/Explanations "+dataName+".csv", index=False)
#
#     # return recall, precision, F1, accuracy, avg_diff, avg_diff_full, spearman_corr, spearman_corr_full
#     return accuracy, avg_diff, avg_diff_full, spearman_corr, spearman_corr_full
