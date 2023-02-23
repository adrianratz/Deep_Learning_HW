import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_DIR = "./tensorflow-datasets/"

# def load_dataset():
#     (ds_train, ds_test), ds_info = tfds.load(
#         'fashion_mnist',
#         split=['train', 'test'],
#         shuffle_files=True,
#         as_supervised=True,
#         with_info=True,
#         data_dir=DATA_DIR
#     )
#     return ds_train, ds_test, ds_info

def load_dataset():
    return tf.keras.datasets.fashion_mnist.load_data()


def split_data(x_train, y_train):
    return train_test_split(x_train, y_train, test_size=0.2)
    

# def normalize_image(image, label):
#     return tf.cast(image, tf.float32) / 255., label


def plot_loss_accuracy(history, title):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(history.history['loss'], label='train')
    ax[0].plot(history.history['val_loss'], label='validation')
    ax[0].set_title(' Loss - ' + title)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(history.history['sparse_categorical_accuracy'], label='train')
    ax[1].plot(history.history['val_sparse_categorical_accuracy'], label='validation')
    ax[1].set_title(' Accuracy - ' + title)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.show()
