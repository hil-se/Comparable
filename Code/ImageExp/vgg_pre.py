import tensorflow as tf
from tensorflow.keras.layers import *


class VGG_Pre:
    def __init__(self, start_size=64, input_shape=(250, 250, 3)):
        base_model = tf.keras.Sequential()
        base_model.add(ZeroPadding2D((1, 1), input_shape=(250, 250, 3)))
        base_model.add(Convolution2D(64, (3, 3), activation='relu'))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(64, (3, 3), activation='relu'))
        base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(128, (3, 3), activation='relu'))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(128, (3, 3), activation='relu'))
        base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(256, (3, 3), activation='relu'))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(256, (3, 3), activation='relu'))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(256, (3, 3), activation='relu'))
        base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        base_model.add(Convolution2D(4096, (7, 7), activation='relu'))
        base_model.add(Dropout(0.5))
        base_model.add(Convolution2D(4096, (1, 1), activation='relu'))
        base_model.add(Dropout(0.5))
        base_model.add(Convolution2D(2622, (1, 1)))
        base_model.add(Flatten())
        base_model.add(Activation('softmax'))

        # pre-trained weights of vgg-face model.
        # you can find it here: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
        # related blog post: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
        base_model.load_weights('../../Data/vgg_face_weights.h5')

        num_of_classes = 1  # this is a regression problem

        # freeze all layers of VGG-Face except last 7 one
        # for layer in base_model.layers[:-7]:
        #     layer.trainable = False

        base_model_output = tf.keras.Sequential()
        base_model_output = Flatten()(base_model.layers[-4].output)
        base_model_output = Dense(num_of_classes)(base_model_output)

        self.model = tf.keras.Model(inputs=base_model.input, outputs=base_model_output)
        self.model.compile(loss='mean_squared_error'
                                     , optimizer=tf.keras.optimizers.Adam())

    def fit(self, X, y, X_val, y_val):
        # pre-trained weights of vgg-face model.
        # you can find it here: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
        # related blog post: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
        #
        # lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1, mode='auto',
        #                                                  min_lr=5e-5)

        # checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoint/attractiveness.keras'
        #                                                   , monitor="val_loss", verbose=1
        #                                                   , save_best_only=True, mode='auto'
        #                                                   )
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        history = self.model.fit(X, y,
                                 callbacks=[early_stopping],
                                 validation_data=(X_val, y_val),
                                 batch_size=128, epochs=100, verbose=1)
        # print(history.history)

    def predict(self, X):
        return self.decision_function(X)

    def decision_function(self, X):
        pred = self.model.predict(X,verbose=0)
        return pred

    def load_model(self, checkpoint_filepath):
        self.model = tf.keras.models.load_model(checkpoint_filepath)
