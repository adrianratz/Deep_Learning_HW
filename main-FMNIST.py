#import libraries 
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Have to use importlib because files are named with hyphens
import importlib
fmnist_util = importlib.import_module("util-FMNIST")
fmnist_model = importlib.import_module("model-FMNIST")

# Split Data into Training and Testing  
(x_train, y_train), (x_test, y_test) = fmnist_util.load_dataset()

# Split data further, peak at data sahpes
x_train, x_val, y_train, y_val = fmnist_util.split_data(x_train, y_train)
print('Training set:   \t Input {0} -\t Output {1}'.format(x_train.shape, y_train.shape))
print('Validation set: \t Input {0} -\t Output {1}'.format(x_val.shape, y_val.shape))
print('Test set:       \t Input {0} -\t Output {1}'.format(x_test.shape, y_test.shape))

# Normalization
# The features in Fashion MNIST [0:255]

# Normalize images from [0:255] -> [0:1] and convert it to 32-bit floating-points
# x_train = x_train / 255.0
# x_train = x_train.astype(np.float32)
# x_train = np.expand_dims(x_train, axis=-1)

# x_val = x_val / 255.0
# x_val = x_val.astype(np.float32)
# x_val = np.expand_dims(x_val, axis=-1)

# x_test = x_test / 255.0
# x_test = x_test.astype(np.float32)
# x_test = np.expand_dims(x_test, axis=-1)

hyperparameters_list = [
    {'num_hidden_layers': 3, 'units': 128, 'dropout_rate': 0.5, 'l2_rate': 0.01, 'learning_rate':0.01},
    {'num_hidden_layers': 3, 'units': 128, 'dropout_rate': 0.2, 'l2_rate': 0.001, 'learning_rate':0.001},
    {'num_hidden_layers': 4, 'units': 256, 'dropout_rate': 0.5, 'l2_rate': 0.01, 'learning_rate':0.01},
    {'num_hidden_layers': 4, 'units': 256, 'dropout_rate': 0.2, 'l2_rate': 0.001, 'learning_rate':0.001},
    {'num_hidden_layers': 3, 'units': 128, 'dropout_rate': 0.2, 'l2_rate': 0.01, 'learning_rate':0.001},
    {'num_hidden_layers': 3, 'units': 128, 'dropout_rate': 0.2, 'l2_rate': 0, 'learning_rate':0.001},
    {'num_hidden_layers': 4, 'units': 256, 'dropout_rate': 0.2, 'l2_rate': 0.01, 'learning_rate':0.001},
    {'num_hidden_layers': 4, 'units': 256, 'dropout_rate': 0.2, 'l2_rate': 0, 'learning_rate':0.001}
]

models = []
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

for i, hyperparameters in tqdm(enumerate(hyperparameters_list), desc="Training models"):
    print(f"Training model {i+1} with hyperparameters: {hyperparameters}")
    model = fmnist_model.create_model(**hyperparameters)
    history = model.fit(x_train,
                        y_train,
                        batch_size=1000,
                        epochs=50,
                        validation_data=(x_val, y_val),
                        verbose=0,
                        callbacks=[callback])
    models.append(model)
    fmnist_util.plot_loss_accuracy(history, f"Model {i+1}")

# find the best model based on validation loss
best_model_index = np.argmin([model.history.history['val_loss'][-1] for model in models])
best_model = models[best_model_index]
print(f"The best model is Model {best_model_index+1} with validation loss: {best_model.history.history['val_loss'][-1]}")

# evaluate the best model on test data
test_loss, test_acc = best_model.evaluate(x_test, y_test)

print(f"Test Loss: {test_loss:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")
