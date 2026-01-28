import tensorflow as tf
import keras
K = tf.keras.backend


def create_encoder(input_size):
    # input = tf.keras.layers.Input(shape=(input_size,))
    # x = tf.keras.layers.Dense(32, activation='relu')(input)
    # output = tf.keras.layers.Dense(1, activation='linear')(input)
    # output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    input_shape = (input_size, 1)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv1D(64, kernel_size=3, activation="relu"),
            keras.layers.Conv1D(64, kernel_size=3, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Flatten(),
            # keras.layers.Conv1D(128, kernel_size=3, activation="relu"),
            # keras.layers.Conv1D(128, kernel_size=3, activation="relu"),
            # keras.layers.GlobalAveragePooling1D(),
            # keras.layers.Dropout(0.5),
            # keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(1, activation="linear"),
        ]
    )

    # return tf.keras.models.Model(inputs=input, outputs=output)
    return model


class DualEncoderAll(tf.keras.Model):
    def __init__(self, encoder, y_true, **kwargs):
        super(DualEncoderAll, self).__init__(**kwargs)
        self.encoder = encoder
        self.y_true = y_true
        self.temperature = 0.05
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, trainable=True):
        encodings_A = self.encoder(features["A"], training=trainable)
        encodings_B = self.encoder(features["B"], training=trainable)
        y = features["Label"]
        self.encoder.trainable = trainable
        return encodings_A, encodings_B, y

    def compute_loss(self, encodings_A, encodings_B, y, sample_weight=None):
        encodings_A = tf.squeeze(encodings_A, axis=-1)
        encodings_B = tf.squeeze(encodings_B, axis=-1)

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
            sw = tf.cast(tf.squeeze(sample_weight, axis=-1) if len(sample_weight.shape) > 1 else sample_weight,
                         tf.float32)
            per_example = per_example * sw
            loss = tf.reduce_sum(per_example) / (tf.reduce_sum(sw) + tf.keras.backend.epsilon())
        else:
            loss = tf.reduce_mean(per_example)

        return loss

    def train_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        sample_weight = x.get("Weights", None) if isinstance(x, dict) else None

        with tf.GradientTape() as tape:
            encodings_A, encodings_B, y = self(x, trainable=True)
            loss = self.compute_loss(encodings_A, encodings_B, y, sample_weight=sample_weight)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # loss is now a scalar, so the Mean metric is happy.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        sample_weight = x.get("Weights", None) if isinstance(x, dict) else None
        encodings_A, encodings_B, y = self(x, trainable=False)
        loss = self.compute_loss(encodings_A, encodings_B, y, sample_weight=sample_weight)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def predict(self, A, B):
        return self.encoder(A) - self.encoder(B)

    def output_grad(self, inputs):
        with tf.GradientTape() as tape:
            loss = self.encoder(inputs)
        grad = tape.gradient(loss, inputs)
        return grad.numpy()[0]

    def score(self, input):
        return self.encoder(input)

    def save(self, path):
        self.encoder.save_weights(path + "_A")
        self.encoder.save_weights(path + "_B")

    def load(self, path):
        self.encoder.load_weights(path + "_A")
        self.encoder.load_weights(path + "_B")
