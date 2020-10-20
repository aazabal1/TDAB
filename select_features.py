import argparse
import logging
import os

import pandas as pd
import numpy as np

import json


LOG = logging.getLogger(__name__)

def select_features(input_args):
    """ Select only appropriate features to input to model """

    with open(input_args.features) as write_file:
        features = json.load(write_file)

    LOG.info(f"Selected features are {features}")
    features.append("DateTime")

    data = pd.read_csv(input_args.dataset,index_col=0)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data = data[features]

    try:
        os.mkdir("data")
    except OSError:
        pass

    path_to_save = f"data/{input_args.save_name}"
    data.to_csv(path_to_save)
    LOG.info(f"Features of interest have been selected and saved to {path_to_save}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=None) # Path to csv dataset
    parser.add_argument("--dataset", default=None) # Rolling mean window
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--save_name", default="data_features_selected.csv") # Name of clean dataset


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

    select_features(args)
