import argparse
import logging
import os


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import load_model

LOG = logging.getLogger(__name__)

def load_test_data(data_folder):
    """ Load X and y test data """

    X_test = np.load(f"data/{data_folder}/X_test.npy")
    y_test = np.load(f"data/{data_folder}/y_test.npy")

    X_noTacking_massTest = np.load(f"data/{data_folder}/X_noTacking_massTest.npy")
    y_noTacking_massTest = np.load(f"data/{data_folder}/y_noTacking_massTest.npy")

    LOG.info(f"Loaded data from data/{data_folder}")
    return X_test, y_test, X_noTacking_massTest, y_noTacking_massTest

def evaluate_model(input_args):
    """ Evaluate performance of selected model """

    # Load test data
    X_test, y_test, X_noTacking_massTest, y_noTacking_massTest = load_test_data(
                                                    input_args.data_folder)

    # Load best model
    if input_args.model_name[-3:]==".h5":
        model = load_model(f"models/{input_args.model_name}")
    else:
        model = load_model(f"models/{input_args.model_name}/best_model.h5")
    LOG.info(f"Loaded best model from models/{input_args.model_name}")

    # Find model accuracy
    _,accuracy = model.evaluate(X_test,y_test)

    # Find model predictions
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred)
    LOG.info("Obtained model predictions")

    LOG.info(classification_report(y_test, y_pred))
    LOG.info(confusion_matrix(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    try:
        os.makedirs(f"results/{input_args.model_name}")
    except OSError:
        pass

    report_df.to_csv(f"results/{input_args.model_name}/results.csv")


    # Get results of other no tacking windows
    y_pred_noTack_test = model.predict(X_noTacking_massTest)
    y_pred_noTack_test = np.round(y_pred_noTack_test)

    LOG.info(classification_report(y_noTacking_massTest, y_pred_noTack_test))
    LOG.info(confusion_matrix(y_noTacking_massTest, y_pred_noTack_test))
    report_noTacking_massTest = classification_report(y_noTacking_massTest,
                    y_pred_noTack_test, output_dict=True)
    report_noTacking_massTest_df = pd.DataFrame(report).transpose()
    report_noTacking_massTest_df.to_csv(f"results/{input_args.model_name}/results_noTack_mass.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default=None) # Path to input X and y train numpy arrays: eg lb_30_hz_30
    parser.add_argument("--model_name", default=None) # Model name eg: model_lb_30_hz_30_16x128_lr0001
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

    evaluate_model(args)
