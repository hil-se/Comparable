import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from sklearn.preprocessing import StandardScaler

import DualEncoder
import SharedDualEncoder

PAIRWISE_DECISION_THRESHOLD = 0.0


def _predict_labels_batch(test, dual_encoder, batch_size=2048):
    dataA = np.asarray(test["A"].tolist())
    dataB = np.asarray(test["B"].tolist())
    if dataA.ndim > 2:
        batch_size = min(batch_size, 4)
    scores = [
        np.asarray(
            dual_encoder.predict(
                dataA[start : start + batch_size],
                dataB[start : start + batch_size],
            )
        ).reshape(-1)
        for start in range(0, len(dataA), batch_size)
    ]
    raw_scores = np.concatenate(scores) if scores else np.array([], dtype=np.float32)
    return np.where(raw_scores < PAIRWISE_DECISION_THRESHOLD, -1, 1).tolist()


def _pair_row(row):
    item = {
        "A": np.asarray(row["A"], dtype=np.float32),
        "B": np.asarray(row["B"], dtype=np.float32),
        "Label": np.float32(row["Label"]),
    }
    if "SA_A" in row.index:
        item["SA_A"] = np.float32(row["SA_A"])
        item["SA_B"] = np.float32(row["SA_B"])
    if "Weights" in row.index:
        item["Weights"] = np.float32(row["Weights"])
    return item


def _pair_signature(df):
    first_row = df.iloc[0]
    signature = {
        "A": tf.TensorSpec(
            shape=np.asarray(first_row["A"], dtype=np.float32).shape,
            dtype=tf.float32,
        ),
        "B": tf.TensorSpec(
            shape=np.asarray(first_row["B"], dtype=np.float32).shape,
            dtype=tf.float32,
        ),
        "Label": tf.TensorSpec(shape=(), dtype=tf.float32),
    }
    if {"SA_A", "SA_B"}.issubset(df.columns):
        signature["SA_A"] = tf.TensorSpec(shape=(), dtype=tf.float32)
        signature["SA_B"] = tf.TensorSpec(shape=(), dtype=tf.float32)
    if "Weights" in df.columns:
        signature["Weights"] = tf.TensorSpec(shape=(), dtype=tf.float32)
    return signature


def _make_pair_dataset(df, batch_size, repeat=True):
    def generator():
        for _, row in df.iterrows():
            yield _pair_row(row)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=_pair_signature(df),
    )
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    return dataset.prefetch(tf.data.AUTOTUNE)


def _fit_array_model(
    model,
    train_x,
    train_y,
    *,
    val_x=None,
    val_y=None,
    epochs=100,
    batch_size=512,
    patience=10,
):
    monitor_metric = "val_loss" if val_x is not None else "loss"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            min_delta=1e-4,
            restore_best_weights=True,
        )
    ]
    fit_kwargs = {
        "x": train_x,
        "y": train_y,
        "epochs": epochs,
        "batch_size": min(batch_size, len(train_x)),
        "verbose": 0,
        "callbacks": callbacks,
    }
    if val_x is not None:
        fit_kwargs["validation_data"] = (val_x, val_y)
    model.fit(**fit_kwargs)
    return model


class SingleEncoderBaseline:
    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor


def _single_encoder_preprocessor(is_binary):
    return StandardScaler() if is_binary else None


def _transform_single_encoder_features(features, preprocessor, fit=False):
    features = np.asarray(features, dtype=np.float32)
    if preprocessor is None:
        return features
    transformed = (
        preprocessor.fit_transform(features)
        if fit
        else preprocessor.transform(features)
    )
    return np.asarray(transformed, dtype=np.float32)


def _shuffle_frame(df, weights=None):
    order = np.random.permutation(len(df))
    shuffled = df.iloc[order].reset_index(drop=True)
    if weights is None:
        return shuffled, None
    return shuffled, np.asarray(weights, dtype=np.float32)[order]


def _batched_scores(inputs, dual_encoder, batch_size=2048):
    inputs = np.asarray(inputs, dtype=np.float32)
    scores = [
        dual_encoder.score(inputs[start : start + batch_size]).numpy().reshape(-1)
        for start in range(0, len(inputs), batch_size)
    ]
    return np.concatenate(scores) if scores else np.array([], dtype=np.float32)


def learn(
    train_data,
    epochs=100,
    validation_data=None,
    y_true=None,
    patience=10,
    batch_size=512,
    shared=False,
    train_weights=None,
    df_name=None,
    fairness_lambda=0.0,
    tabular_encoder_type="cnn",
    output_activation="linear",
):
    is_scut = df_name is not None and "scut" in df_name.lower()
    if is_scut:
        batch_size = min(batch_size, 2)

    train_frame = train_data.copy()
    if train_weights is not None:
        train_frame["Weights"] = np.asarray(train_weights, dtype=np.float32)

    train_dataset = _make_pair_dataset(train_frame, batch_size)
    steps_per_epoch = int(np.ceil(len(train_data) / batch_size))
    y_true = np.array([] if y_true is None else y_true)
    input_size = train_dataset.element_spec["A"].shape[1]

    if shared:
        encoder = SharedDualEncoder.create_encoder(
            input_size=input_size,
            df_name=df_name,
            output_activation=output_activation,
            tabular_encoder_type=tabular_encoder_type,
        )
        dual_encoder = SharedDualEncoder.DualEncoderAll(
            encoder,
            y_true=y_true,
            fairness_lambda=fairness_lambda,
        )
    else:
        encoder_A = DualEncoder.create_encoder(input_size=input_size)
        encoder_B = DualEncoder.create_encoder(input_size=input_size)
        dual_encoder = DualEncoder.DualEncoderAll(
            encoder_A,
            encoder_B,
            y_true=y_true,
        )
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
            val_dataset = _make_pair_dataset(validation_data, batch_size)
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
    train_pixels = np.asarray(train_pixels, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    if val_pixels is not None:
        val_pixels = np.asarray(val_pixels, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

    model = SharedDualEncoder.create_encoder(input_size=None, df_name="scut_baseline")
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4),
        loss="mse",
    )
    return _fit_array_model(
        model,
        train_pixels,
        y_train,
        val_x=val_pixels,
        val_y=y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )


def predict_scut_vggface_baseline(model, pixels, batch_size=4):
    pixels = np.asarray(pixels, dtype=np.float32)
    return model.predict(pixels, batch_size=batch_size, verbose=0).reshape(-1)


def train_single_encoder_baseline(
    train_features,
    y_train,
    is_binary,
    output_activation=None,
    val_features=None,
    y_val=None,
    epochs=100,
    batch_size=512,
    patience=10,
    tabular_encoder_type="cnn",
):
    preprocessor = _single_encoder_preprocessor(is_binary)
    train_features = _transform_single_encoder_features(
        train_features,
        preprocessor,
        fit=True,
    )
    y_train = np.asarray(y_train, dtype=np.float32)
    if val_features is not None:
        val_features = _transform_single_encoder_features(
            val_features,
            preprocessor,
        )
        y_val = np.asarray(y_val, dtype=np.float32)

    if output_activation is None:
        output_activation = "sigmoid" if is_binary else "linear"
    model = SharedDualEncoder.create_encoder(
        input_size=train_features.shape[1],
        df_name="tabular_baseline",
        output_activation=output_activation,
        tabular_encoder_type=tabular_encoder_type,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy" if is_binary else "mse",
    )
    return SingleEncoderBaseline(
        model=_fit_array_model(
            model,
            train_features,
            y_train,
            val_x=val_features,
            val_y=y_val,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
        ),
        preprocessor=preprocessor,
    )


def predict_single_encoder_baseline(model, features, batch_size=2048):
    baseline = (
        model
        if isinstance(model, SingleEncoderBaseline)
        else SingleEncoderBaseline(model=model)
    )
    features = _transform_single_encoder_features(
        features,
        baseline.preprocessor,
    )
    return baseline.model.predict(features, batch_size=batch_size, verbose=0).reshape(-1)


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
    tabular_encoder_type="cnn",
    output_activation="linear",
):
    train, train_weights = _shuffle_frame(train, train_weights)
    if val is not None:
        val = val.reset_index(drop=True).copy()
        if val_weights is not None:
            val["Weights"] = np.asarray(val_weights, dtype=np.float32)
    return learn(
        train,
        epochs=epochs,
        validation_data=val,
        y_true=y_true,
        shared=shared,
        train_weights=train_weights,
        df_name=df_name,
        fairness_lambda=fairness_lambda,
        tabular_encoder_type=tabular_encoder_type,
        output_activation=output_activation,
    )


def predict(test, dual_encoder):
    return _predict_labels_batch(test, dual_encoder)


def test_model(test, dual_encoder):
    labels = test["Label"].tolist()
    predictions = _predict_labels_batch(test, dual_encoder)
    return predictions, evaluate(labels, predictions)


def evaluate(y_true, y_pred):
    labels = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    tp = np.sum((labels == 1) & (predictions == 1))
    fp = np.sum((labels != 1) & (predictions == 1))
    tn = np.sum((labels != 1) & (predictions != 1))
    fn = np.sum((labels == 1) & (predictions != 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        (2 * recall * precision) / (recall + precision)
        if recall + precision
        else 0.0
    )
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    return recall, precision, f1, accuracy


def generateLists(test_dataset, dual_encoder):
    real_list = pd.DataFrame(
        {"id": test_dataset["indexA"].values, "Score": test_dataset["Score"].values}
    )
    pred_list = pd.DataFrame(
        {
            "id": test_dataset["indexA"].values,
            "Score": _batched_scores(np.stack(test_dataset["A"].values), dual_encoder),
        }
    )
    spearmanr, sp_pvalue = stats.spearmanr(real_list["Score"], pred_list["Score"])
    pearsonr, p_pvalue = stats.pearsonr(real_list["Score"], pred_list["Score"])
    return spearmanr, sp_pvalue, pearsonr, p_pvalue


def evaluateLists(realList, predList):
    real_ids = realList["id"].tolist()
    pred_positions = {value: idx for idx, value in enumerate(predList["id"].tolist())}
    diffs = np.array(
        [idx - pred_positions[value] for idx, value in enumerate(real_ids)]
    )
    ln = len(real_ids)
    spearman_corr = round(1 - ((6 * np.sum(diffs**2)) / (ln * ((ln * ln) - 1))), 3)
    avg_diff = round(np.mean(np.abs(diffs)), 3)
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
