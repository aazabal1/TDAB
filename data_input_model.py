import argparse
import logging
import os

import pandas as pd
import numpy as np
import datetime

from src.scalers import RNN_Transform_Wrap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


LOG = logging.getLogger(__name__)


def data_loading(input_args):
    """ Loads clean data, splits into into windows defined by the user, splits
        into test and train and scales data
        Arguments: dataset - clean dataset to prepare to input to model
                   lookback - # of time steps to look back before prediction (ie: LSTM # timesteps)
                   horizon - # of time steps to predict in the future (ie: if 1, predict next timestep, if 10, predict 10 seconds later)
                    """

    # Load dataset with features of interest
    df = pd.read_csv(input_args.dataset,index_col=0)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # Find max time and remove lookback and horizon from it
    max_t = df["DateTime"].max()
    df_window = df[df["DateTime"]<=(max_t-datetime.timedelta(0,input_args.lookback+input_args.horizon))]


    # Split data into windows of size "lookback"
    LOG.info("Splitting data into windows")
    start_times = pd.to_datetime(df_window.iloc[::input_args.lookback, :]["DateTime"]).reset_index(drop=True)#.values

    # Find actual tacking value of each lookback window
    # Tacking value = tacking at the horizon timestep
    tack_df = pd.DataFrame({"StartTime":[0]*len(start_times),"Tacking":[0]*len(start_times)})
    for i, times in enumerate(start_times):
        tack_df.loc[i,"StartTime"] = times
        tack = df[df["DateTime"]==(times+datetime.timedelta(0,input_args.lookback+input_args.horizon))]["Tacking"].values[0]
        tack_df.loc[i,"Tacking"] = tack

    # Get windows where horizon time is tacking
    tacking_yes = tack_df[tack_df["Tacking"]==1].reset_index(drop=True)
    start_tacking_yes = tacking_yes["StartTime"]

    LOG.info(f"Found {len(start_tacking_yes)} windows with tacking")
    # Get windows where horizon time is not tacking. Choose n number of windows
    # depeding on desired class balance
    n_noTack = round(len(tacking_yes)*input_args.class_balance/(1-input_args.class_balance))
    tacking_no = tack_df[tack_df["Tacking"]==0].sample(n_noTack).reset_index(drop=True)
    start_tacking_no = tacking_no["StartTime"]

    # Select remaining windows with no tacking to be used as part of test data
    tacking_no_testSet = tack_df[(tack_df["Tacking"]==0)&(-tack_df["StartTime"].isin(start_tacking_no))].reset_index(drop=True)
    start_tacking_no_test = tacking_no_testSet["StartTime"]

    tacking_all_df = pd.concat([tacking_yes,tacking_no]).reset_index(drop=True)
    LOG.info(f"Final dataset has {len(start_tacking_yes)+len(start_tacking_no)} windows")
    # Create X and y datasets. Start with windows with tacking
    # X dimensions are: [# windows, # timesteps=lookback, # features]. This is reshaped
    # later in the code
    X_ones = np.zeros((input_args.lookback+1,len(df.columns)-2,len(start_tacking_yes)))
    y_ones = np.ones(len(start_tacking_yes))

    for i,times in enumerate(start_tacking_yes):
        case = df[(df["DateTime"]>=times)
                  &(df["DateTime"]<=times+datetime.timedelta(0,input_args.lookback))
                 ].drop(columns=["DateTime","Tacking"]).to_numpy()
        X_ones[:,:,i] = case

    # Reshape X for proper order of axes
    X_ones = X_ones.transpose((2, 0, 1))

    # Same process for windows with no tacking
    X_zeros = np.zeros((input_args.lookback+1,len(df.columns)-2,len(start_tacking_no)))
    y_zeros = np.zeros(len(start_tacking_no))

    for i,times in enumerate(start_tacking_no):
        case = df[(df["DateTime"]>=times)
                  &(df["DateTime"]<=times+datetime.timedelta(0,input_args.lookback))
                 ].drop(columns=["DateTime","Tacking"]).to_numpy()
        X_zeros[:,:,i] = case

    X_zeros = X_zeros.transpose((2, 0, 1))

    # Combine both datasets together
    X = np.vstack((X_zeros,X_ones))
    y = np.hstack((y_zeros,y_ones))

    # Split data into train and test
    LOG.info("Split data into train and test datasets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=input_args.test_split, random_state=42)

    # Scale X data
    x_scaler = RNN_Transform_Wrap(MinMaxScaler)
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    # Repeat above process for remaning windows with no tacking. Used later for test
    X_noTack_massTest = np.zeros((input_args.lookback+1,len(df.columns)-2,len(start_tacking_no_test)))
    y_noTack_massTest = np.zeros(len(start_tacking_no_test))

    for i,times in enumerate(start_tacking_no_test):
        case = df[(df["DateTime"]>=times)
                  &(df["DateTime"]<=times+datetime.timedelta(0,input_args.lookback))
                 ].drop(columns=["DateTime","Tacking"]).to_numpy()
        X_noTack_massTest[:,:,i] = case

    X_noTack_massTest = X_noTack_massTest.transpose((2, 0, 1))
    X_noTack_massTest = x_scaler.transform(X_noTack_massTest)

    try:
        os.makedirs(f"data/lb_{input_args.lookback}_hz_{input_args.horizon}")
    except OSError:
        pass

    # Save data as numpy arrays
    LOG.info(f"Saving data as numpy arrays in folder data/lb_{input_args.lookback}_hz_{input_args.horizon}")
    np.save(f"data/lb_{input_args.lookback}_hz_{input_args.horizon}/X_train.npy",X_train)
    np.save(f"data/lb_{input_args.lookback}_hz_{input_args.horizon}/X_test.npy",X_test)
    np.save(f"data/lb_{input_args.lookback}_hz_{input_args.horizon}/y_train.npy",y_train)
    np.save(f"data/lb_{input_args.lookback}_hz_{input_args.horizon}/y_test.npy",y_test)
    np.save(f"data/lb_{input_args.lookback}_hz_{input_args.horizon}/X_noTacking_massTest.npy",X_noTack_massTest)
    np.save(f"data/lb_{input_args.lookback}_hz_{input_args.horizon}/y_noTacking_massTest.npy",y_noTack_massTest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None) # Path to dataset to scale and split
    parser.add_argument("--lookback", default=None, type=int) # Train lookback
    parser.add_argument("--horizon", default=None, type=int) # Predict horizon
    parser.add_argument("--test_split", default=0.2) # Train test split
    parser.add_argument("--class_balance", default=0.5) # Class balance split
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

    data_loading(args)
