import tensorflow as tf
import keras
from pathlib import Path
from tensorflow.keras.layers import *

K = tf.keras.backend
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "Data"


def create_encoder(input_size, df_name):

    if df_name is not None and "scut" in df_name:
        base_model = tf.keras.Sequential()
        base_model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        base_model.add(Convolution2D(64, (3, 3), activation="relu"))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(64, (3, 3), activation="relu"))
        base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(128, (3, 3), activation="relu"))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(128, (3, 3), activation="relu"))
        base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(256, (3, 3), activation="relu"))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(256, (3, 3), activation="relu"))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(256, (3, 3), activation="relu"))
        base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(512, (3, 3), activation="relu"))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(512, (3, 3), activation="relu"))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(512, (3, 3), activation="relu"))
        base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(512, (3, 3), activation="relu"))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(512, (3, 3), activation="relu"))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(512, (3, 3), activation="relu"))
        base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        base_model.add(Convolution2D(4096, (7, 7), activation="relu"))
        base_model.add(Dropout(0.5))
        base_model.add(Convolution2D(4096, (1, 1), activation="relu"))
        base_model.add(Dropout(0.5))
        base_model.add(Convolution2D(2622, (1, 1)))
        base_model.add(Flatten())
        base_model.add(Activation("softmax"))

        # Pre-trained weights of VGG-Face model.
        base_model.load_weights(str(DATA_DIR / "vgg_face_weights.h5"))

        # Full fine-tuning: train all backbone layers.
        for layer in base_model.layers:
            layer.trainable = True

        # Keras 3 may not populate `base_model.input` until the model is called once.
        _ = base_model(tf.keras.Input(shape=(224, 224, 3)))

        # Use the penultimate feature map (layer -4), then add a regression head.
        x = base_model.layers[-4].output
        x = Flatten()(x)
        output = Dense(1, activation="linear")(x)
        model = tf.keras.Model(inputs=base_model.inputs, outputs=output)

    else:
        # Tabular (non-SCUT) encoder: basic 1D CNN over the feature vector.
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(input_size,)),
                keras.layers.Reshape((input_size, 1)),
                keras.layers.Conv1D(
                    filters=16, kernel_size=3, padding="same", activation="relu"
                ),
                keras.layers.Conv1D(
                    filters=32, kernel_size=3, padding="same", activation="relu"
                ),
                keras.layers.GlobalMaxPooling1D(),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

    # return tf.keras.models.Model(inputs=input, outputs=output)
    return model


class DualEncoderAll(tf.keras.Model):
    def __init__(self, encoder, y_true, fairness_lambda=0.0, **kwargs):
        super(DualEncoderAll, self).__init__(**kwargs)
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
        """
        Convert encoder output to one scalar per sample.
        Handles both shape [batch, 1] and higher-dimensional outputs (e.g. [batch, 2622]).
        """
        encoding = tf.cast(encoding, tf.float32)
        encoding = tf.reshape(encoding, [tf.shape(encoding)[0], -1])
        return tf.reduce_mean(encoding, axis=1)

    def compute_loss(
        self, encodings_A, encodings_B, y, sample_weight=None, sa_a=None, sa_b=None
    ):
        encodings_A = self._to_scalar(encodings_A)
        encodings_B = self._to_scalar(encodings_B)
        pred = encodings_A - encodings_B
        y = tf.cast(tf.squeeze(y, axis=-1) if len(y.shape) > 1 else y, tf.float32)

        per_example = tf.abs(y - pred)

        # Hinge loss
        # loss = tf.math.maximum(0.0, 1.0 - (y * pred))

        # Absolute hinge-like loss
        # loss = tf.math.abs(1 - (y * pred))

        # Averaged hinge loss
        # loss_A = tf.math.maximum(0.0, 1 - (y*encodings_A))
        # loss_B = tf.math.maximum(0.0, 1 - (y*encodings_B))
        # loss = (loss_A+loss_B)/2

        # Binary cross-entropy loss
        # bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # loss = tf.math.abs(bce(y, pred))

        # Mean-squared error loss
        # mse = tf.keras.losses.MeanSquaredError()
        # loss = tf.math.abs(mse(y, pred))
        if sample_weight is not None:
            sw = tf.cast(
                tf.squeeze(sample_weight, axis=-1)
                if len(sample_weight.shape) > 1
                else sample_weight,
                tf.float32,
            )
            per_example = per_example * sw
            loss = tf.reduce_sum(per_example) / (
                tf.reduce_sum(sw) + tf.keras.backend.epsilon()
            )
        else:
            loss = tf.reduce_mean(per_example)

        if self.fairness_lambda > 0 and sa_a is not None and sa_b is not None:
            sa_a = tf.cast(
                tf.squeeze(sa_a, axis=-1) if len(sa_a.shape) > 1 else sa_a, tf.float32
            )
            sa_b = tf.cast(
                tf.squeeze(sa_b, axis=-1) if len(sa_b.shape) > 1 else sa_b, tf.float32
            )
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
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        sample_weight = x.get("Weights", None) if isinstance(x, dict) else None
        sa_a = x.get("SA_A", None) if isinstance(x, dict) else None
        sa_b = x.get("SA_B", None) if isinstance(x, dict) else None

        with tf.GradientTape() as tape:
            encodings_A, encodings_B, y, sa_a_out, sa_b_out = self(x, trainable=True)
            sa_a = sa_a if sa_a is not None else sa_a_out
            sa_b = sa_b if sa_b is not None else sa_b_out
            loss = self.compute_loss(
                encodings_A,
                encodings_B,
                y,
                sample_weight=sample_weight,
                sa_a=sa_a,
                sa_b=sa_b,
            )

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # loss is now a scalar, so the Mean metric is happy.
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
