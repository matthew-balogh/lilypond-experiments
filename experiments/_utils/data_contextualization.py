import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from _utils.constants import RANDOM_SEED
from ucimlrepo import fetch_ucirepo

def __load_dataset():
    dataset = fetch_ucirepo(id=193)
    data = dataset.data.features
    target = dataset.data.targets["NSP"]
    return data, target

def __prepare_data(data, target, verb=True, random_seed=RANDOM_SEED):
    if verb: print(f"The dataset has the following shape: {data.shape} / {target.shape}")

    if verb:
        print("The data points grouped by label:")
        print(target.value_counts())

    # historical / holdout split

    X = data.copy()
    y = target.copy().astype(int)

    X_hist, X_holdout, y_hist, y_holdout =  train_test_split(X, y, test_size=0.5, random_state=random_seed, shuffle=True, stratify=y)

    if verb:
            print(f"Historical set: {X_hist.shape} / {y_hist.shape}")
            print("\t > The data points grouped by label:")
            print(y_hist.value_counts())

            print(f"Holdout set: {X_holdout.shape} / {y_holdout.shape}")
            print("\t > The data points grouped by label:")
            print(y_holdout.value_counts())

    X_hist_normal = X_hist[y_hist == 1]
    X_hist_suspect = X_hist[y_hist == 2]
    X_hist_abnormal = X_hist[y_hist == 3]

    X_holdout_normal = X_holdout[y_holdout == 1]
    X_holdout_suspect = X_holdout[y_holdout == 2]
    X_holdout_abnormal = X_holdout[y_holdout == 3]

    return X_hist_normal, X_hist_suspect, X_hist_abnormal, X_holdout_normal, X_holdout_suspect, X_holdout_abnormal

def __load_prepare_data(random_seed=RANDOM_SEED):
    data, target = __load_dataset()
    return __prepare_data(data, target, random_seed=random_seed)

DATASET_NAMES = ["Cardio"]
DATASET_GETTERS = lambda random_seed: map(lambda x: (lambda: __load_prepare_data(random_seed=random_seed)), DATASET_NAMES)
DATASET_GETTERS_FULL = map(lambda x: (lambda: __load_dataset()), DATASET_NAMES)
