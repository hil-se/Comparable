import pandas as pd
import tensorflow as tf

image_path = "../../Data/ImageExp/images/"


def load_scut(file="../data/train.csv"):
    def retrievePixels(path):
        img = tf.keras.utils.load_img(image_path + path, target_size=(250, 250), grayscale=False)
        x = tf.keras.utils.img_to_array(img)
        return x

    data = pd.read_csv(file)
    data['pixels'] = data['Filename'].apply(retrievePixels)
    return data
