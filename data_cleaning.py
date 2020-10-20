import argparse
import logging
import os

import pandas as pd
import numpy as np
import datetime

LOG = logging.getLogger(__name__)

def load_data(input_data):
    """ Load data from raw csv dataset and return dataframe """

    data = pd.read_csv(input_data)
    LOG.info(f"Dataset read from {input_data}")
    return data

def clean_data(input_args):
    """ Clean raw data (remove duplicates, NaNs, smooths data and
        selects proper start time). Saves clean data to csv file for later use"""
    data = load_data(input_args.dataset)

    LOG.info("Starting to clean dataset")
    # Remove duplicates and fill NaN with previous value
    data = data.fillna(method='ffill')
    data = data.drop_duplicates()

    # Split data between discrete and continous variables
    booleans = data[["ModePilote","Tacking","DateTime"]].set_index("DateTime")
    data_noB = data.drop(columns=["ModePilote","Tacking"]).set_index("DateTime")

    # Apply rolling mean to continous variables
    data_rw = data_noB.rolling(input_args.rolling_mean).mean()

    # Join discrete and continous variables with rolling mean
    data_df = pd.concat([data_rw, booleans], axis=1)
    data_df = data_df.dropna()
    data_df.reset_index(level=0, inplace=True)

    # Select data from start time identified graphically
    data_start = data_df[data_df["DateTime"]>= input_args.start_time].reset_index(drop=True)
    data_start['DateTime'] = pd.to_datetime(data_start['DateTime'])

    # Identify rows with repeated datetimes (these are rows where the seconds
    # have been erroneously saved)
    data_start['DateTime'] = pd.to_datetime(data_start['DateTime'])
    data_start["ShiftDT"] = data_start["DateTime"].shift(1)
    shifts = data_start["DateTime"]-data_start["ShiftDT"]
    data_start.drop(columns=["ShiftDT"],inplace=True)
    rep_time = shifts[shifts=="0s"]
    idx_rep = rep_time.index

    # Add one second to every row with a 1 second larg
    for idx in idx_rep:
        data_start.loc[idx,"DateTime"] = data_start.loc[idx,"DateTime"]+datetime.timedelta(0,1)

    try:
        os.mkdir("data")
    except OSError:
        pass

    path_to_save = f"data/{input_args.save_name}"
    data_start.to_csv(path_to_save)
    LOG.info(f"Dataset has been cleaned and saved to {path_to_save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None) # Path to csv dataset
    parser.add_argument("--rolling_mean", default=60) # Rolling mean window
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--start_time", default="2019-04-14 09:46:40") # YYYY-MM-DD hh-mm-ss
    parser.add_argument("--save_name", default="cleaned_data.csv") # Name of clean dataset


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

    clean_data(args)
