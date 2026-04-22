from pathlib import Path

import keras
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Conv1D,
    Convolution2D,
    Dense,
    Dropout,
    Flatten,
    GlobalMaxPooling1D,
    MaxPooling2D,
    Reshape,
    ZeroPadding2D,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "Data"
TABULAR_ENCODER_TYPES = {"cnn", "linear"}


def _is_scut_model(df_name):
    return bool(df_name and "scut" in df_name.lower())


def _build_scut_encoder(output_activation):
    base_model = tf.keras.Sequential(
        [
            ZeroPadding2D((1, 1), input_shape=(224, 224, 3)),
            Convolution2D(64, (3, 3), activation="relu"),
            ZeroPadding2D((1, 1)),
            Convolution2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2), strides=(2, 2)),
            ZeroPadding2D((1, 1)),
            Convolution2D(128, (3, 3), activation="relu"),
            ZeroPadding2D((1, 1)),
            Convolution2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2), strides=(2, 2)),
            ZeroPadding2D((1, 1)),
            Convolution2D(256, (3, 3), activation="relu"),
            ZeroPadding2D((1, 1)),
            Convolution2D(256, (3, 3), activation="relu"),
            ZeroPadding2D((1, 1)),
            Convolution2D(256, (3, 3), activation="relu"),
            MaxPooling2D((2, 2), strides=(2, 2)),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, (3, 3), activation="relu"),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, (3, 3), activation="relu"),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, (3, 3), activation="relu"),
            MaxPooling2D((2, 2), strides=(2, 2)),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, (3, 3), activation="relu"),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, (3, 3), activation="relu"),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, (3, 3), activation="relu"),
            MaxPooling2D((2, 2), strides=(2, 2)),
            Convolution2D(4096, (7, 7), activation="relu"),
            Dropout(0.5),
            Convolution2D(4096, (1, 1), activation="relu"),
            Dropout(0.5),
            Convolution2D(2622, (1, 1)),
            Flatten(),
            Activation("softmax"),
        ]
    )
    base_model.load_weights(str(DATA_DIR / "vgg_face_weights.h5"))
    for layer in base_model.layers:
        layer.trainable = True

    _ = base_model(tf.keras.Input(shape=(224, 224, 3)))
    x = Flatten()(base_model.layers[-4].output)
    output = Dense(1, activation=output_activation)(x)
    return tf.keras.Model(inputs=base_model.inputs, outputs=output)


def _build_tabular_encoder(input_size, output_activation):
    return keras.Sequential(
        [
            keras.layers.Input(shape=(input_size,)),
            Reshape((input_size, 1)),
            Conv1D(filters=16, kernel_size=3, padding="same", activation="relu"),
            Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
            GlobalMaxPooling1D(),
            Dense(16, activation="relu"),
            Dense(1, activation=output_activation),
        ]
    )


def _build_linear_tabular_encoder(input_size, output_activation):
    return keras.Sequential(
        [
            keras.layers.Input(shape=(input_size,)),
            Dense(1, activation=output_activation),
        ]
    )


def _normalize_tabular_encoder_type(encoder_type):
    normalized = str(encoder_type).strip().lower()
    if normalized not in TABULAR_ENCODER_TYPES:
        valid = ", ".join(sorted(TABULAR_ENCODER_TYPES))
        raise ValueError(
            f"Unknown tabular encoder type '{encoder_type}'. Valid options: {valid}"
        )
    return normalized


def create_encoder(
    input_size,
    df_name,
    output_activation="linear",
    tabular_encoder_type="cnn",
):
    if _is_scut_model(df_name):
        return _build_scut_encoder(output_activation)

    tabular_encoder_type = _normalize_tabular_encoder_type(tabular_encoder_type)
    if tabular_encoder_type == "linear":
        return _build_linear_tabular_encoder(input_size, output_activation)
    return _build_tabular_encoder(input_size, output_activation)


class DualEncoderAll(tf.keras.Model):
    def __init__(self, encoder, y_true, fairness_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.y_true = y_true
        self.fairness_lambda = fairness_lambda
        self.temperature = 0.05
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, trainable=True):
        encodings_A = self.encoder(features["A"], training=trainable)
        encodings_B = self.encoder(features["B"], training=trainable)
        y = features["Label"]
        sa_a = features.get("SA_A", None)
        sa_b = features.get("SA_B", None)
        self.encoder.trainable = trainable
        return encodings_A, encodings_B, y, sa_a, sa_b

    @staticmethod
    def _to_scalar(encoding):
        encoding = keras.ops.cast(encoding, "float32")
        encoding = keras.ops.reshape(encoding, (keras.ops.shape(encoding)[0], -1))
        return keras.ops.mean(encoding, axis=1)

    @staticmethod
    def _flat_float(value):
        return tf.cast(tf.reshape(value, (-1,)), tf.float32)

    @staticmethod
    def _unpack_inputs(data):
        return data[0] if isinstance(data, tuple) else data

    def _compute_pairwise_loss(
        self, encodings_A, encodings_B, y, sample_weight=None, sa_a=None, sa_b=None
    ):
        encodings_A = self._to_scalar(encodings_A)
        encodings_B = self._to_scalar(encodings_B)
        pred = encodings_A - encodings_B
        y = self._flat_float(y)

        per_example = tf.abs(y - pred)
        if sample_weight is not None:
            sw = self._flat_float(sample_weight)
            per_example = per_example * sw
            loss = tf.reduce_sum(per_example) / (
                tf.reduce_sum(sw) + tf.keras.backend.epsilon()
            )
        else:
            loss = tf.reduce_mean(per_example)

        if self.fairness_lambda > 0 and sa_a is not None and sa_b is not None:
            sa_a = self._flat_float(sa_a)
            sa_b = self._flat_float(sa_b)
            cross_group = tf.not_equal(sa_a, sa_b)
            within_group = tf.equal(sa_a, sa_b)

            cross_count = tf.reduce_sum(tf.cast(cross_group, tf.float32))
            within_count = tf.reduce_sum(tf.cast(within_group, tf.float32))

            def _safe_group_mean(mask):
                masked = tf.boolean_mask(per_example, mask)
                return tf.reduce_mean(masked)

            fairness_penalty = tf.cond(
                tf.logical_and(cross_count > 0, within_count > 0),
                lambda: tf.abs(
                    _safe_group_mean(cross_group) - _safe_group_mean(within_group)
                ),
                lambda: tf.constant(0.0, dtype=tf.float32),
            )
            loss = loss + self.fairness_lambda * fairness_penalty

        return loss

    def train_step(self, data):
        x = self._unpack_inputs(data)
        sample_weight = x.get("Weights") if isinstance(x, dict) else None
        sa_a = x.get("SA_A") if isinstance(x, dict) else None
        sa_b = x.get("SA_B") if isinstance(x, dict) else None

        with tf.GradientTape() as tape:
            encodings_A, encodings_B, y, sa_a_out, sa_b_out = self(x, trainable=True)
            sa_a = sa_a if sa_a is not None else sa_a_out
            sa_b = sa_b if sa_b is not None else sa_b_out
            loss = self._compute_pairwise_loss(
                encodings_A,
                encodings_B,
                y,
                sample_weight=sample_weight,
                sa_a=sa_a,
                sa_b=sa_b,
            )

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x = self._unpack_inputs(data)
        sample_weight = x.get("Weights") if isinstance(x, dict) else None
        sa_a = x.get("SA_A") if isinstance(x, dict) else None
        sa_b = x.get("SA_B") if isinstance(x, dict) else None

        encodings_A, encodings_B, y, sa_a_out, sa_b_out = self(x, trainable=False)
        sa_a = sa_a if sa_a is not None else sa_a_out
        sa_b = sa_b if sa_b is not None else sa_b_out
        loss = self._compute_pairwise_loss(
            encodings_A,
            encodings_B,
            y,
            sample_weight=sample_weight,
            sa_a=sa_a,
            sa_b=sa_b,
        )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def predict(self, A, B):
        pred = self._to_scalar(self.encoder(A)) - self._to_scalar(self.encoder(B))
        return tf.expand_dims(pred, axis=-1)

    def output_grad(self, inputs):
        with tf.GradientTape() as tape:
            loss = self.encoder(inputs)
        grad = tape.gradient(loss, inputs)
        return grad.numpy()[0]

    def score(self, input):
        score = self._to_scalar(self.encoder(input))
        return tf.expand_dims(score, axis=-1)

    def save(self, path):
        self.encoder.save_weights(path + "_A")
        self.encoder.save_weights(path + "_B")

    def load(self, path):
        self.encoder.load_weights(path + "_A")
        self.encoder.load_weights(path + "_B")
