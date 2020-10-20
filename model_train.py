import argparse
import logging
import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

LOG = logging.getLogger(__name__)

def load_train_data(data_folder):
    """ Load X and y train datasets """

    X_train = np.load(f"data/{data_folder}/X_train.npy")
    y_train = np.load(f"data/{data_folder}/y_train.npy")
    LOG.info(f"Loaded data from data/{data_folder}")
    return X_train, y_train

def build_model(input_args):
    """ Build neural network, define hyperparameters and tran model """

    # Load train data
    X_train, y_train = load_train_data(input_args.data_folder)

    # Input layer
    input_layer = Input(shape=(X_train.shape[1],X_train.shape[2]),name="input_layer")

    # LSTM stacked layers
    x = Bidirectional(LSTM(input_args.input_layer_size,activation="tanh",
    recurrent_activation="sigmoid",return_sequences=True, recurrent_dropout=input_args.dropout),
                  merge_mode="ave")(input_layer)
    x = Dropout(input_args.dropout)(x)
    x = Bidirectional(LSTM(input_args.output_layer_size,activation="tanh",
    recurrent_activation="sigmoid",return_sequences=False, recurrent_dropout=input_args.dropout),
                  merge_mode="ave")(x)

    # Dense and output layers
    x = Dropout(input_args.dropout)(x)
    x = Dense(input_args.output_layer_size,activation="relu")(x)
    x = Dropout(input_args.dropout)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(input_layer, outputs, name="model")
    LOG.info(model.summary())

    # Compile model
    optimizer = Adam(learning_rate=input_args.learning_rate)

    try:
        os.makedirs(f"models/{input_args.model_name}")
    except OSError:
        pass

    checkpoint = ModelCheckpoint(f"models/{input_args.model_name}/best_model.h5", monitor='val_accuracy', save_best_only=True,
    save_weights_only=False, mode='max')
    model.compile(optimizer, loss="binary_crossentropy", metrics= 'accuracy' )

    # Train model
    LOG.info("Starting to train model")
    history = model.fit(X_train, y_train, batch_size = input_args.batch_size, epochs = input_args.n_epochs,
    validation_split = input_args.val_split, callbacks = [checkpoint])

    LOG.info(f"Best model saved at models/{input_args.model_name}/best_model.h5")

    # Plot evolution of training
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f"models/{input_args.model_name}/train_loss.png")
    plt.close()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f"models/{input_args.model_name}/train_accuracy.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default=None) # Path to input X and y train numpy arrays: eg lb_30_hz_30
    parser.add_argument("--model_name", default=None) # Model name eg: model_lb_30_hz_30_16x128_lr0001
    parser.add_argument("--input_layer_size", default=None, type=int) # Input layer size
    parser.add_argument("--output_layer_size", default=None, type=int) # Output layer size
    parser.add_argument("--learning_rate", default=0.0001) # Learning rate
    parser.add_argument("--batch_size", default=1) # Batch size
    parser.add_argument("--dropout", default=None,type=float) # Batch size
    parser.add_argument("--n_epochs", default=40) # Number of epochs
    parser.add_argument("--val_split", default=0.1) # Validation split
    parser.add_argument("--log_level", default="INFO")


    args = parser.parse_args()

    # Set up logging format
    LOG = logging.getLogger()
    LOG.setLevel(args.log_level)
    sh = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-20s %(levelname)-8s %(message)s")
    sh.setFormatter(formatter)
    LOG.addHandler(sh)
    LOG.info("#####################")
    LOG.info("Start running script")
    LOG.info("#####################")

    build_model(args)
