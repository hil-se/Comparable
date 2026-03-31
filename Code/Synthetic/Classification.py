import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

import DualEncoder
import SharedDualEncoder

PAIRWISE_DECISION_THRESHOLD = 0.0


def _predict_labels_batch(test, dual_encoder, batch_size=2048):
    """Batch inference to avoid per-row Python/TensorFlow overhead."""
    dataA = np.asarray(test["A"].tolist())
    dataB = np.asarray(test["B"].tolist())
    if dataA.ndim > 2:
        batch_size = min(batch_size, 4)
    scores = []
    for start in range(0, len(dataA), batch_size):
        end = min(start + batch_size, len(dataA))
        batch_scores = dual_encoder.predict(dataA[start:end], dataB[start:end])
        scores.append(np.asarray(batch_scores).reshape(-1))
    raw_scores = np.concatenate(scores) if scores else np.array([], dtype=np.float32)
    return np.where(raw_scores < PAIRWISE_DECISION_THRESHOLD, -1, 1).tolist()


def learn(
    train_data,
    epochs=100,
    validation_data=None,
    y_true=None,
    patience=10,
    batch_size=512,
    shared=False,
    train_weights=None,
    val_weights=None,
    df_name=None,
    fairness_lambda=0.0,
):
    # SCUT image pairs are large; use a tiny batch and stream samples to avoid OOM.
    is_scut = df_name is not None and "scut" in df_name.lower()
    if is_scut:
        batch_size = min(batch_size, 2)

    first_a = np.asarray(train_data["A"].iloc[0], dtype=np.float32)
    first_b = np.asarray(train_data["B"].iloc[0], dtype=np.float32)

    def _row_generator():
        if train_weights is None:
            for _, row in train_data.iterrows():
                item = {
                    "A": np.asarray(row["A"], dtype=np.float32),
                    "B": np.asarray(row["B"], dtype=np.float32),
                    "Label": np.float32(row["Label"]),
                }
                if "SA_A" in row and "SA_B" in row:
                    item["SA_A"] = np.float32(row["SA_A"])
                    item["SA_B"] = np.float32(row["SA_B"])
                yield item
        else:
            for idx, row in train_data.iterrows():
                item = {
                    "A": np.asarray(row["A"], dtype=np.float32),
                    "B": np.asarray(row["B"], dtype=np.float32),
                    "Label": np.float32(row["Label"]),
                    "Weights": np.float32(train_weights[idx]),
                }
                if "SA_A" in row and "SA_B" in row:
                    item["SA_A"] = np.float32(row["SA_A"])
                    item["SA_B"] = np.float32(row["SA_B"])
                yield item

    output_signature = {
        "A": tf.TensorSpec(shape=first_a.shape, dtype=tf.float32),
        "B": tf.TensorSpec(shape=first_b.shape, dtype=tf.float32),
        "Label": tf.TensorSpec(shape=(), dtype=tf.float32),
    }
    if train_weights is not None:
        output_signature["Weights"] = tf.TensorSpec(shape=(), dtype=tf.float32)
    if "SA_A" in train_data.columns and "SA_B" in train_data.columns:
        output_signature["SA_A"] = tf.TensorSpec(shape=(), dtype=tf.float32)
        output_signature["SA_B"] = tf.TensorSpec(shape=(), dtype=tf.float32)

    train_dataset = tf.data.Dataset.from_generator(
        _row_generator, output_signature=output_signature
    )
    steps_per_epoch = int(np.ceil(len(train_data) / batch_size))
    train_dataset = train_dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    if y_true is None:
        y_true = []

    if shared:
        encoder = SharedDualEncoder.create_encoder(
            input_size=train_dataset.element_spec["A"].shape[1], df_name=df_name
        )
        dual_encoder = SharedDualEncoder.DualEncoderAll(
            encoder, y_true=np.array(y_true), fairness_lambda=fairness_lambda
        )
    else:
        encoder_A = DualEncoder.create_encoder(
            input_size=train_dataset.element_spec["A"].shape[1]
        )
        encoder_B = DualEncoder.create_encoder(
            input_size=train_dataset.element_spec["A"].shape[1]
        )
        dual_encoder = DualEncoder.DualEncoderAll(
            encoder_A, encoder_B, y_true=np.array(y_true)
        )
    # Full fine-tuning on pretrained SCUT backbone is more stable with a smaller LR.
    learning_rate = 1e-4 if is_scut else 1e-3
    optimizer = (
        tf.keras.optimizers.SGD(learning_rate=learning_rate)
        if is_scut
        else tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )
    dual_encoder.compile(
        optimizer=optimizer,
        jit_compile=not is_scut,
    )
    monitor_metric = "val_loss" if validation_data is not None else "loss"
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        min_delta=1e-4,
        restore_best_weights=True,
    )
    fit_kwargs = {
        "x": train_dataset,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "verbose": 2 if shared else 0,
        "callbacks": [early_stopping],
    }
    if validation_data is not None:
        if isinstance(validation_data, pd.DataFrame):
            val_first_a = np.asarray(validation_data["A"].iloc[0], dtype=np.float32)
            val_first_b = np.asarray(validation_data["B"].iloc[0], dtype=np.float32)

            def _val_row_generator():
                for _, row in validation_data.iterrows():
                    item = {
                        "A": np.asarray(row["A"], dtype=np.float32),
                        "B": np.asarray(row["B"], dtype=np.float32),
                        "Label": np.float32(row["Label"]),
                    }
                    if "SA_A" in validation_data.columns and "SA_B" in validation_data.columns:
                        item["SA_A"] = np.float32(row["SA_A"])
                        item["SA_B"] = np.float32(row["SA_B"])
                    if "Weights" in validation_data.columns:
                        item["Weights"] = np.float32(row["Weights"])
                    yield item

            val_signature = {
                "A": tf.TensorSpec(shape=val_first_a.shape, dtype=tf.float32),
                "B": tf.TensorSpec(shape=val_first_b.shape, dtype=tf.float32),
                "Label": tf.TensorSpec(shape=(), dtype=tf.float32),
            }
            if "SA_A" in validation_data.columns and "SA_B" in validation_data.columns:
                val_signature["SA_A"] = tf.TensorSpec(shape=(), dtype=tf.float32)
                val_signature["SA_B"] = tf.TensorSpec(shape=(), dtype=tf.float32)
            if "Weights" in validation_data.columns:
                val_signature["Weights"] = tf.TensorSpec(shape=(), dtype=tf.float32)

            val_dataset = tf.data.Dataset.from_generator(
                _val_row_generator, output_signature=val_signature
            )
            val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            fit_kwargs["validation_data"] = val_dataset
            fit_kwargs["validation_steps"] = int(
                np.ceil(len(validation_data) / batch_size)
            )
        else:
            fit_kwargs["validation_data"] = validation_data

    dual_encoder.fit(**fit_kwargs)

    return dual_encoder


def train_scut_vggface_baseline(
    train_pixels,
    y_train,
    val_pixels=None,
    y_val=None,
    epochs=100,
    batch_size=2,
    patience=10,
):
    """
    Train a non-comparative SCUT baseline: VGG-Face regressor on single images.
    """
    train_pixels = np.asarray(train_pixels, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    if train_pixels.ndim != 4:
        raise ValueError("train_pixels must have shape [N, H, W, C].")
    if len(train_pixels) != len(y_train):
        raise ValueError("train_pixels and y_train must have the same length.")
    if val_pixels is not None or y_val is not None:
        if val_pixels is None or y_val is None:
            raise ValueError("val_pixels and y_val must be provided together.")
        val_pixels = np.asarray(val_pixels, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)
        if len(val_pixels) != len(y_val):
            raise ValueError("val_pixels and y_val must have the same length.")

    model = SharedDualEncoder.create_encoder(input_size=None, df_name="scut_baseline")
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4),
        loss="mse",
    )
    monitor_metric = "val_loss" if val_pixels is not None else "loss"
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        min_delta=1e-4,
        restore_best_weights=True,
    )
    fit_kwargs = {
        "x": train_pixels,
        "y": y_train,
        "epochs": epochs,
        "batch_size": min(batch_size, len(train_pixels)),
        "verbose": 0,
        "callbacks": [early_stopping],
    }
    if val_pixels is not None:
        fit_kwargs["validation_data"] = (val_pixels, y_val)
    model.fit(**fit_kwargs)
    return model


def predict_scut_vggface_baseline(model, pixels, batch_size=4):
    pixels = np.asarray(pixels, dtype=np.float32)
    return model.predict(pixels, batch_size=batch_size, verbose=0).reshape(-1)


def train_single_encoder_baseline(
    train_features,
    y_train,
    is_binary,
    val_features=None,
    y_val=None,
    epochs=100,
    batch_size=512,
    patience=10,
):
    """
    Train a non-comparative tabular baseline with a single encoder.
    """
    train_features = np.asarray(train_features, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    if train_features.ndim != 2:
        raise ValueError("train_features must have shape [N, D].")
    if len(train_features) != len(y_train):
        raise ValueError("train_features and y_train must have the same length.")
    if val_features is not None or y_val is not None:
        if val_features is None or y_val is None:
            raise ValueError("val_features and y_val must be provided together.")
        val_features = np.asarray(val_features, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)
        if len(val_features) != len(y_val):
            raise ValueError("val_features and y_val must have the same length.")

    model = SharedDualEncoder.create_encoder(
        input_size=train_features.shape[1], df_name="tabular_baseline"
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy" if is_binary else "mse",
    )
    monitor_metric = "val_loss" if val_features is not None else "loss"
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        min_delta=1e-4,
        restore_best_weights=True,
    )
    fit_kwargs = {
        "x": train_features,
        "y": y_train,
        "epochs": epochs,
        "batch_size": min(batch_size, len(train_features)),
        "verbose": 0,
        "callbacks": [early_stopping],
    }
    if val_features is not None:
        fit_kwargs["validation_data"] = (val_features, y_val)
    model.fit(**fit_kwargs)
    return model


def predict_single_encoder_baseline(model, features, batch_size=2048):
    features = np.asarray(features, dtype=np.float32)
    return model.predict(features, batch_size=batch_size, verbose=0).reshape(-1)


def train_model(
    train,
    val,
    y_true,
    epochs=100,
    shared=False,
    train_weights=None,
    val_weights=None,
    df_name=None,
    fairness_lambda=0.0,
):
    if train_weights is not None:
        train_weights = np.asarray(train_weights, dtype=np.float32)
        if len(train_weights) != len(train):
            raise ValueError("train_weights length must match training data length.")
        perm = np.random.permutation(len(train))
        train = train.iloc[perm].reset_index(drop=True)
        train_weights = train_weights[perm]
    else:
        train = train.sample(frac=1).reset_index(drop=True)
    if val is not None:
        val = val.reset_index(drop=True).copy()
        if val_weights is not None:
            val_weights = np.asarray(val_weights, dtype=np.float32)
            if len(val_weights) != len(val):
                raise ValueError("val_weights length must match validation data length.")
            val["Weights"] = val_weights
    dual_encoder = learn(
        train,
        epochs=epochs,
        validation_data=val,
        y_true=y_true,
        shared=shared,
        train_weights=train_weights,
        val_weights=val_weights,
        df_name=df_name,
        fairness_lambda=fairness_lambda,
    )
    return dual_encoder


def predict(test, dual_encoder):
    return _predict_labels_batch(test, dual_encoder)


def test_model(test, dual_encoder):
    labels = test["Label"].tolist()
    predictions = _predict_labels_batch(test, dual_encoder)
    return predictions, evaluate(labels, predictions)


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


def generateLists(test_dataset, dual_encoder):
    realList = []
    predList = []
    for _, row in test_dataset.iterrows():
        idName = row["indexA"]
        datapoint = np.array(row["A"])
        real_score = row["Score"]
        datapoint = np.expand_dims(datapoint, axis=0)
        pred_score = dual_encoder.score(datapoint)
        pred_score = pred_score.numpy()[0][0].item()
        realList.append({"id": idName, "Score": real_score})
        predList.append({"id": idName, "Score": pred_score})
    realList = pd.DataFrame(realList)
    predList = pd.DataFrame(predList)

    [spearmanr, sp_pvalue] = stats.spearmanr(realList["Score"], predList["Score"])
    [pearsonr, p_pvalue] = stats.pearsonr(realList["Score"], predList["Score"])
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
        diff += abs(i - j)
        sum_d += (i - j) * (i - j)
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
