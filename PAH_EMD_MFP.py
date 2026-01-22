import pickle
import statistics
import deepchem as dc
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization
from random import seed
from random import shuffle
from scipy.stats import pearsonr
from deepchem.models import AttentiveFPModel
import os

# Set random seed to ensure reproducibility
seed(13)

# Load data
path = "./Fold_Predictions/"
with open("3.2_CH_Cleaner ALL_High_PAHs Dataset.pickle", "rb") as f:
    d = pickle.load(f)
smiles = d["smiles"]
sequences = d["sequences"]

# Shuffle data
dataset = list(zip(smiles, sequences))
shuffle(dataset)

# Data analysis to ensure duplicate molecules do not cross folds
single_occurrence_molecules = [x for x in dataset if list(d["smiles"]).count(x[0]) <= 1]
multiple_occurrence_molecules = [x for x in dataset if x[0] not in [h[0] for h in single_occurrence_molecules]]

folds = {}
fold_size = len(single_occurrence_molecules) // 5
for i in range(1, 6):
    folds[i] = single_occurrence_molecules[((i - 1) * fold_size):(i * fold_size)]

multiple_occurrence_molecules += single_occurrence_molecules[(5 * fold_size):]

current_fold = 0
while len(multiple_occurrence_molecules) > 0:
    current_fold %= 5
    current_fold += 1
    current_molecule = multiple_occurrence_molecules[0]
    while current_molecule[0] in [h[0] for h in multiple_occurrence_molecules]:
        folds[current_fold].append(
            multiple_occurrence_molecules.pop(
                [h[0] for h in multiple_occurrence_molecules].index(current_molecule[0])
            )
        )

# Print the size of each fold
for i in range(1, 6):
    print(len(folds[i]))


# Helper functions
def normalize(s):
    max_val = max(s)
    scale = 1 / max_val
    if max_val == 0:
        scale = 0
    return [j * scale for j in s]


def floor_out(x):
    return [j if j > 0.01 else 0 for j in x]


def normal_many(x):
    return np.array([floor_out(normalize(j)) for j in x])


# EMD loss function
def emd_loss(y_true, y_pred):
    """
    Earth Mover's Distance (EMD) loss function.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if len(y_true.shape) == 1:
        y_true = tf.expand_dims(y_true, axis=0)
    if len(y_pred.shape) == 1:
        y_pred = tf.expand_dims(y_pred, axis=0)

    # Normalize
    y_true = y_true / tf.reduce_sum(y_true, axis=1, keepdims=True)
    y_pred = y_pred / tf.reduce_sum(y_pred, axis=1, keepdims=True)

    cum_true = tf.cumsum(y_true, axis=1)
    cum_pred = tf.cumsum(y_pred, axis=1)

    emd = tf.reduce_mean(tf.reduce_sum(tf.abs(cum_true - cum_pred), axis=1))
    return emd


def pearson_first(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def wrapped_pearson_correlation(y_true, y_pred):
    y = tf.py_function(func=pearson_first, inp=[y_true, y_pred], Tout=tf.float32)
    return y


def clean(arr):
    arr = list(map(float, arr))
    return [item for item in arr if not np.isnan(item)]


# Create dataset splits
dataset_splits = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
for i in range(1, 6):
    test = folds[i]
    train = []
    for x in range(1, 6):
        if x != i:
            train += folds[x]

    dataset_splits[i]["test_smiles"] = [j[0] for j in test]
    dataset_splits[i]["test_y"] = normal_many([j[1] for j in test])
    dataset_splits[i]["train_smiles"] = [j[0] for j in train]
    dataset_splits[i]["train_y"] = normal_many([j[1] for j in train])


# Use Morgan fingerprints
featurizer = dc.feat.CircularFingerprint(radius=2, size=1024, chiral=False, features=False)
for i in range(1, 6):
    dataset_splits[i]["test_x"] = featurizer.featurize(dataset_splits[i]["test_smiles"])
    dataset_splits[i]["train_x"] = featurizer.featurize(dataset_splits[i]["train_smiles"])


# Five-fold cross-validation
for i in range(1, 6):
    fpmodel = Sequential()
    fpmodel.add(Dense(4096, input_dim=1024))
    fpmodel.add(BatchNormalization())
    fpmodel.add(Dropout(0.1))
    fpmodel.add(Dense(2048, activation="relu"))
    fpmodel.add(BatchNormalization())
    fpmodel.add(Dropout(0.1))
    fpmodel.add(Dense(1024, activation="relu"))
    fpmodel.add(Dense(180, activation="sigmoid"))

    # Use EMD as the loss function
    fpmodel.compile(loss=emd_loss, optimizer="Adam")

    fpmodel.fit(
        dataset_splits[i]["train_x"],
        dataset_splits[i]["train_y"],
        batch_size=64,
        epochs=150,
        verbose=0
    )

    # Collect evaluation metrics
    morgan_predictions = fpmodel.predict(dataset_splits[i]["test_x"])
    total_r2, count = 0, 0
    total_emd = 0

    fp_r2s = []
    for x in range(len(morgan_predictions)):
        current_r2 = wrapped_pearson_correlation(
            normalize(morgan_predictions[x]),
            dataset_splits[i]["test_y"][x]
        )
        current_emd = emd_loss(dataset_splits[i]["test_y"][x], morgan_predictions[x])

        total_r2 += 0 if np.isnan(current_r2) else current_r2
        total_emd += current_emd

        fp_r2s.append(current_r2)
        count += 1

    current_fold_loss = round(float(total_r2 / count), 5)
    current_fold_emd_loss = round(float(total_emd / count), 5)

    print("R2 Loss for fold", i, ":", current_fold_loss)
    print("EMD Loss for fold", i, ":", current_fold_emd_loss)

    clean_fp_r2s = clean(list(map(float, fp_r2s)))
    print(
        "I", i,
        statistics.mean(clean_fp_r2s),
        statistics.median(clean_fp_r2s),
        statistics.stdev(clean_fp_r2s)
    )

    # ========================
    # Save predictions (including smiles and true values)
    # ========================
    results = []
    for s, p, t in zip(
        dataset_splits[i]["test_smiles"],
        morgan_predictions,
        dataset_splits[i]["test_y"]
    ):
        results.append({
            "smiles": s,
            "pred": p,
            "true": t
        })

    fold_predictions_path = path + f"ALL_High_MFP_Fold_{i}_results.pickle"
    save_dir = os.path.dirname(fold_predictions_path)
    os.makedirs(save_dir, exist_ok=True)
    with open(fold_predictions_path, 'wb') as handle:
        pickle.dump(results, handle)

    print(f"ALL_High_Fold {i} results saved to {fold_predictions_path}")
